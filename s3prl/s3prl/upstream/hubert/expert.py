# Copyright (c) Facebook, Inc. All Rights Reserved

# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/hubert/expert.py ]
#   Synopsis     [ the HuBERT wrapper ]
#   Author       [ Kushal Lakhotia ]
"""*********************************************************************************************"""

import torch
import logging
import torch.nn.functional as F
from pathlib import Path
# For S3Delta
from scipy.optimize import fsolve
import numpy as np

from torch.nn.utils.rnn import pad_sequence

from ..interfaces import UpstreamBase
from .convert import load_converted_model

SAMPLE_RATE = 16000
EXAMPLE_SEC = 5

logger = logging.getLogger(__name__)


class UpstreamExpert(UpstreamBase):
    def __init__(self, ckpt, **kwargs):
        super().__init__(**kwargs)
        model, task_cfg = load_converted_model(ckpt, **kwargs)
        self.model = model
        self.task_cfg = task_cfg

        self.model.feature_grad_mult = 0.0
        self.model.encoder.layerdrop = 0.0

        if len(self.hooks) == 0:
            module_name = "self.model.encoder.layers"
            for module_id in range(len(eval(module_name))):
                self.add_hook(
                    f"{module_name}[{module_id}]",
                    lambda input, output: input[0].transpose(0, 1),
                )
            self.add_hook("self.model.encoder", lambda input, output: output[0])

            def postprocess(xs):
                names, hiddens = zip(*xs)
                unpad_len = min([hidden.size(1) for hidden in hiddens])
                hiddens = [hidden[:, :unpad_len, :] for hidden in hiddens]
                return list(zip(names, hiddens))

            self.hook_postprocess = postprocess

        self.last_zeta = 0.

    def get_downsample_rates(self, key: str) -> int:
        return 320

    def reduce_tau(self):
        for layer in self.model.encoder.layers:
            layer.adapterswitch.reduce_tau()
    
    def compute_zeta(self, max_num_param: float, tau: float, eps=1e-7):
        flatten_alpha = torch.stack(self.model.all_alpha(), dim=0).view(-1).tolist()
        alphas = np.array(flatten_alpha)
        param_nums = np.array(self.model.param_nums)
        def func(zeta):
            x = (alphas - zeta) / tau
            x[x < -200] = -200
            sigmoid_shifted_alphas = 1 / (1 + np.exp(-x))
            ret = (sigmoid_shifted_alphas * param_nums).sum() - \
                max_num_param * (1 - eps)
            return ret
        zeta_res = fsolve(func, self.last_zeta)[0]
        
        self.last_zeta = zeta_res

        return zeta_res

    def compute_shifted_sigmoid(self, max_num_param: float, tau: float, eps=1e-7, use_last=True):
        if use_last:
            curr_zeta = self.last_zeta
        else:
            curr_zeta = self.compute_zeta(max_num_param, tau, eps)
        all_alphas = torch.stack(self.model.all_alpha(), dim=0)
        x = (all_alphas - curr_zeta) / tau
        x[x < -200] = -200

        sigmoid_x = torch.sigmoid(x)
        dp = torch.zeros_like(sigmoid_x)
        dp[sigmoid_x == 1] = -eps
        dp[sigmoid_x == 0] = eps
        sigmoid_x = sigmoid_x + dp

        norm = sigmoid_x.sum().detach() / sigmoid_x.sum()
        for idx, layer in enumerate(self.model.encoder.layers):
            layer.adapterswitch.p = sigmoid_x[idx] * norm
            # print(f'layer_{idx}.p = {layer.adapterswitch.p}')

    def set_hard_forward_structure(self, max_num_param: float, baseline=None):
        if baseline:
            for layer_idx, layer in enumerate(self.model.encoder.layers):
                layer.adapterswitch.fixed_idx = layer.adapterswitch.paths
            return
        # shape: [num_layers, num_path]
        all_alpha = torch.stack(self.model.all_alpha(), dim=0)
        all_alpha = [
            (layer_idx, path_idx, element.item()) 
            for layer_idx, logits in enumerate(all_alpha) 
            for path_idx, element in enumerate(logits)
        ]
        all_alpha = sorted(all_alpha, key=lambda x:x[2], reverse=True)
        # Use an adapter with the highest prob when the current total number of parameters are still within the budget.
        curr_num_param = 0
        selected_path = [[] for _ in range(len(self.model.encoder.layers))]
        for alpha in all_alpha:
            layer_idx, path_idx = alpha[0], alpha[1]
            num_param = self.model.encoder.layers[layer_idx].delta_list[path_idx].num_parameter
            if curr_num_param + num_param <= max_num_param:
                selected_path[layer_idx].append(path_idx)
                curr_num_param += num_param
        print(f'[upstream/hubert/expert.py]: current number of parameters: {curr_num_param}')
        
        for layer_idx, layer in enumerate(self.model.encoder.layers):
            layer.adapterswitch.fixed_idx = sorted(selected_path[layer_idx])

    def aux_loss(self):
        loss = 0
        for layer in self.model.encoder.layers:
            loss += layer.aux_loss()
        return loss

    def set_stage(self, stage: int):
        assert stage == 1 or stage == 2, "stage most be 1 or 2"
        for layer in self.model.encoder.layers:
            layer.adapterswitch.config.stage = stage

    def sample_gumbel(self):
        # Sample gumbel noise for switch
        for layer in self.model.encoder.layers:
            layer.adapterswitch.sample_gumbel()

    def sample_uniform(self):
        # Sample uniform noise for switch
        for layer in self.model.encoder.layers:
            layer.adapterswitch.sample_uniform()

    # For the second-order approximation in DARTS algorithm
    def use_virtual(self):
        for layer in self.model.encoder.layers:
            layer.use_virtual()

    def use_default(self):
        for layer in self.model.encoder.layers:
            layer.use_default()

    @property
    def get_layers(self):
        return self.model.encoder.layers
    
    @property
    def get_named_parameters(self):
        return self.model.named_parameters()

    def forward(self, wavs):
        if self.task_cfg.normalize:
            wavs = [F.layer_norm(wav, wav.shape) for wav in wavs]

        device = wavs[0].device
        wav_lengths = torch.LongTensor([len(wav) for wav in wavs]).to(device)
        wav_padding_mask = ~torch.lt(
            torch.arange(max(wav_lengths)).unsqueeze(0).to(device),
            wav_lengths.unsqueeze(1),
        )
        padded_wav = pad_sequence(wavs, batch_first=True)

        features, feat_padding_mask = self.model.extract_features(
            padded_wav,
            padding_mask=wav_padding_mask,
            mask=None,
        )

        # This forward function only does the model forward
        # The return dict is then handled by UpstreamBase's hooks


class LegacyUpstreamExpert(UpstreamBase):
    def __init__(self, ckpt, **kwargs):
        super().__init__(**kwargs)
        logger.warning("Use the legacy expert for HuBERT which depends on fairseq")
        import fairseq

        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt])
        self.model = model[0]
        self.task = task

        self.model.feature_grad_mult = 0.0
        self.model.encoder.layerdrop = 0.0

        if len(self.hooks) == 0:
            module_name = "self.model.encoder.layers"
            for module_id in range(len(eval(module_name))):
                self.add_hook(
                    f"{module_name}[{module_id}]",
                    lambda input, output: input[0].transpose(0, 1),
                )
            self.add_hook("self.model.encoder", lambda input, output: output[0])

            def postprocess(xs):
                names, hiddens = zip(*xs)
                unpad_len = min([hidden.size(1) for hidden in hiddens])
                hiddens = [hidden[:, :unpad_len, :] for hidden in hiddens]
                return list(zip(names, hiddens))

            self.hook_postprocess = postprocess

    def get_downsample_rates(self, key: str) -> int:
        return 320

    def forward(self, wavs):
        if self.task_cfg.normalize:
            wavs = [F.layer_norm(wav, wav.shape) for wav in wavs]

        device = wavs[0].device
        wav_lengths = torch.LongTensor([len(wav) for wav in wavs]).to(device)
        wav_padding_mask = ~torch.lt(
            torch.arange(max(wav_lengths)).unsqueeze(0).to(device),
            wav_lengths.unsqueeze(1),
        )
        padded_wav = pad_sequence(wavs, batch_first=True)

        features, feat_padding_mask = self.model.extract_features(
            padded_wav,
            padding_mask=wav_padding_mask,
            mask=None,
        )

        # This forward function only does the model forward
        # The return dict is then handled by UpstreamBase's hooks


class LegacyUpstreamExpert(UpstreamBase):
    def __init__(self, ckpt, **kwargs):
        super().__init__(**kwargs)
        logger.warning("Use the legacy expert for HuBERT which depends on fairseq")
        import fairseq

        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt])
        self.model = model[0]
        self.model.feature_grad_mult = 0.0
        self.task = task

        if len(self.hooks) == 0:
            module_name = "self.model.encoder.layers"
            for module_id in range(len(eval(module_name))):
                self.add_hook(
                    f"{module_name}[{module_id}]",
                    lambda input, output: input[0].transpose(0, 1),
                )
            self.add_hook("self.model.encoder", lambda input, output: output[0])

            def postprocess(xs):
                names, hiddens = zip(*xs)
                unpad_len = min([hidden.size(1) for hidden in hiddens])
                hiddens = [hidden[:, :unpad_len, :] for hidden in hiddens]
                return list(zip(names, hiddens))

            self.hook_postprocess = postprocess

    def get_downsample_rates(self, key: str) -> int:
        return 320

    def forward(self, wavs):
        if self.task.cfg.normalize:
            wavs = [F.layer_norm(wav, wav.shape) for wav in wavs]

        device = wavs[0].device
        wav_lengths = torch.LongTensor([len(wav) for wav in wavs]).to(device)
        wav_padding_mask = ~torch.lt(
            torch.arange(max(wav_lengths)).unsqueeze(0).to(device),
            wav_lengths.unsqueeze(1),
        )
        padded_wav = pad_sequence(wavs, batch_first=True)

        features, feat_padding_mask = self.model.extract_features(
            padded_wav,
            padding_mask=wav_padding_mask,
            mask=None,
        )

        # This forward function only does the model forward
        # The return dict is then handled by UpstreamBase's hooks
