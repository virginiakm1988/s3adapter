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

    def get_downsample_rates(self, key: str) -> int:
        return 320

    def reduce_tau(self):
        for layer in self.model.encoder.layers:
            layer.adapterswitch.reduce_tau()
    
    def compute_zeta(self, prev_zeta: float, max_num_param: float, tau: float, eps=1e-7):
        alphas = np.array(self.model.all_alpha)
        param_nums = np.array(self.model.param_nums)

        def func(zeta):
            x = (alphas - zeta) / tau
            x[x < -200] = -200
            sigmoid_shifted_alphas = 1 / (1 + np.exp(-x))
            ret = (sigmoid_shifted_alphas * param_nums).sum() - \
                max_num_param * (1 - eps)
            return ret
        zeta_res = fsolve(func, prev_zeta)[0]
        
        # setup zeta for all switch
        for layer in self.model.encoder.layers():
            layer.adapterswitch.zeta = zeta_res

        return zeta_res

    def compute_shifted_sigmoid(self, prev_zeta, max_num_param, tau, eps=1e-7):
        curr_zeta = self.compute_zeta(prev_zeta, max_num_param, tau, eps)

        all_alphas = torch.cat(self.model.all_alpha, dim=0)
        x = (all_alphas - curr_zeta) / tau
        x[x < -200] = -200

        sigmoid_x = torch.sigmoid(x)
        dp = torch.zeros_like(sigmoid_x)
        dp[sigmoid_x == 1] = -eps
        dp[sigmoid_x == 0] = eps
        sigmoid_x = sigmoid_x + dp

        norm = sigmoid_x.detach().sum / sigmoid_x.sum()
        for layer in self.model.encoder.layers:
            layer.adapterswitch.p = torch.sigmoid((layer.adapterswitch.switch_logits - curr_zeta) / tau) * norm

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
