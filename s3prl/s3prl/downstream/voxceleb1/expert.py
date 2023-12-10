# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ expert.py ]
#   Synopsis     [ the phone linear downstream wrapper ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import math
import torch
import random
import pathlib
import wandb
#-------------#
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed import is_initialized
from torch.nn.utils.rnn import pad_sequence
#-------------#
from ..model import *
from .dataset import SpeakerClassifiDataset
from argparse import Namespace
from pathlib import Path


class DownstreamExpert(nn.Module):
    """
    Used to handle downstream-specific operations
    eg. downstream forward, metric computation, contents to log
    """

    def __init__(self, upstream_dim, downstream_expert, expdir, **kwargs):
        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.downstream = downstream_expert
        self.datarc = downstream_expert['datarc']
        self.modelrc = downstream_expert['modelrc']
        self.expdir = expdir

        # AdapterConfig
        if 'adapterConfig' in kwargs:
            self.adapterConfig = kwargs['adapterConfig']
        else:
            self.adapterConfig = None
            print("[voxceleb1/expert.py] 51: No Adapter Config")
        self.switch_ratio = 0.0

        root_dir = Path(self.datarc['file_path'])

        self.train_dataset_full = SpeakerClassifiDataset('train', root_dir, self.datarc['meta_data'], self.datarc['max_timestep'])
        self.train_dataset = None
        self.switch_dataset = None
        self.dev_dataset = SpeakerClassifiDataset('dev', root_dir, self.datarc['meta_data'])
        self.test_dataset = SpeakerClassifiDataset('test', root_dir, self.datarc['meta_data'])
        
        model_cls = eval(self.modelrc['select'])
        model_conf = self.modelrc.get(self.modelrc['select'], {})
        self.projector = nn.Linear(upstream_dim, self.modelrc['projector_dim'])
        self.model = model_cls(
            input_dim = self.modelrc['projector_dim'],
            output_dim = self.train_dataset_full.speaker_num,
            **model_conf,
        )
        self.curr_projector = self.projector
        self.curr_model = self.model
        if 'do_virtual' in kwargs and kwargs['do_virtual']:
            self.virtual_projector = nn.Linear(upstream_dim, self.modelrc['projector_dim'])
            self.virtual_model = model_cls(
                input_dim = self.modelrc['projector_dim'],
                output_dim = self.train_dataset_full.speaker_num,
                **model_conf,
            )
            for virtual_p in self.virtual_projector.parameters():
                setattr(virtual_p, '__is_virtual__', True)
            for virtual_p in self.virtual_model.parameters():
                setattr(virtual_p, '__is_virtual__', True)
        
        self.objective = nn.CrossEntropyLoss()
        self.register_buffer('best_score', torch.zeros(1))

    def _get_train_dataloader(self, dataset):
        sampler = DistributedSampler(dataset) if is_initialized() else None
        return DataLoader(
            dataset, batch_size=self.datarc['train_batch_size'], 
            shuffle=(sampler is None), sampler=sampler,
            num_workers=self.datarc['num_workers'],
            collate_fn=dataset.dataset.collate_fn
        )

    def _get_eval_dataloader(self, dataset):
        return DataLoader(
            dataset, batch_size=self.datarc['eval_batch_size'],
            shuffle=False, num_workers=self.datarc['num_workers'],
            collate_fn=dataset.collate_fn
        )

    def get_train_dataloader(self):
        return self._get_train_dataloader(self.train_dataset)

    # DataLoader for stage 1 switch training
    def get_switch_dataloader(self):
        return None if len(self.switch_dataset) == 0 else self._get_train_dataloader(self.switch_dataset)

    def get_dev_dataloader(self):
        return self._get_eval_dataloader(self.dev_dataset)

    def get_test_dataloader(self):
        return self._get_eval_dataloader(self.test_dataset)
    
    # Interface
    def copy_params(self):
        self.virtual_projector.load_state_dict(self.projector.state_dict())
        self.virtual_model.load_state_dict(self.model.state_dict())

    # Interface
    def use_virtual(self):
        self.copy_params()
        self.curr_projector = self.virtual_projector
        self.curr_model = self.virtual_model
        
    # Interface
    def use_default(self):
        self.curr_projector = self.projector
        self.curr_model = self.model

    # Interface
    def get_dataloader(self, split, mode=None, epoch=None):
        assert self.adapterConfig != None, "adapterConfig is none!"
        if split == 'train':
            # reset the switch dataset ratio
            self.switch_ratio = self.adapterConfig.adapter.switch.ratio
            # devide the dataset
            self.train_dataset, self.switch_dataset = \
                torch.utils.data.random_split(self.train_dataset_full, [1 - self.switch_ratio, self.switch_ratio])

            # Cast Subset to Dataset
            # self.train_dataset, self.switch_dataset = self.train_dataset.dataset, self.switch_dataset.dataset
            # return two dataloader
            return {
                'train': self.get_train_dataloader(),
                'switch': self.get_switch_dataloader()
            }
        
        return eval(f'self.get_{split}_dataloader')()

    # Interface
    def forward(self, mode, features, labels, filenames, records, **kwargs):
        device = features[0].device
        features_len = torch.IntTensor([len(feat) for feat in features]).to(device=device)
        features = pad_sequence(features, batch_first=True)
        features = self.curr_projector(features)
        predicted, _ = self.curr_model(features, features_len)

        labels = torch.LongTensor(labels).to(features.device)
        loss = self.objective(predicted, labels)

        predicted_classid = predicted.max(dim=-1).indices
        if kwargs.get('record', False):
            records['acc'] += (predicted_classid == labels).view(-1).cpu().float().tolist()
            records["loss"].append(loss.item())
            records['filename'] += filenames
            records['predict_speaker'] += SpeakerClassifiDataset.label2speaker(predicted_classid.cpu().tolist())
            records['truth_speaker'] += SpeakerClassifiDataset.label2speaker(labels.cpu().tolist())
        if kwargs.get('return_predicted', False):
            return loss, predicted
        return loss

    # interface
    def log_records(self, mode, records, logger, global_step, **kwargs):
        wandb.define_metric("dev-acc", summary="max")
        wandb.define_metric("train-acc", summary="max")
        
        save_names = []
        results = {} # results update to wandb
        for key in ["acc", "loss"]:
            average = torch.FloatTensor(records[key]).mean().item()
            logger.add_scalar(
                f'voxceleb1/{mode}-{key}',
                average,
                global_step=global_step
            )
            with open(Path(self.expdir) / "log.log", 'a') as f:
                if key == 'acc':
                    print(f"{mode} {key}: {average}")
                    f.write(f'{mode} at step {global_step}: {average}\n')
                    if mode == 'dev' and average > self.best_score:
                        self.best_score = torch.ones(1).to(self.best_score.device) * average
                        f.write(f'New best on {mode} at step {global_step}: {average}\n')
                        save_names.append(f'{mode}-best.ckpt')
            results.update({f'{mode}-{key}': average})

        if mode == 'train': 
            average_aux_loss = torch.FloatTensor(records['aux_loss']).mean().item() if len(records['aux_loss']) > 0 else 0
            logger.add_scalar(
                f'voxceleb1/{mode}-aux_loss', average_aux_loss, global_step=global_step
            )
            results.update({f'{mode}-aux_loss': average_aux_loss})
            #print(f'aux_loss {average_aux_loss}')

            total_loss = results[f'{mode}-loss'] + average_aux_loss
            logger.add_scalar(
                f'voxceleb1/{mode}-total_loss', total_loss, global_step=global_step
            )
            results.update({f'{mode}-total_loss': total_loss})

            if len(records['grad_norm']):
                avg_grad_norm = torch.FloatTensor(records['grad_norm']).mean().item()
                results.update({f'{mode}-grad_norm': avg_grad_norm})
                logger.add_scalar(
                    f'voxceleb1/{mode}-grad_norm', avg_grad_norm, global_step=global_step
                )

            if len(records['kl_loss']):
                avg_kl_loss = torch.FloatTensor(records['kl_loss']).mean().item()
                results.update({f'{mode}-kl_loss': avg_kl_loss})
                logger.add_scalar(
                    f'voxceleb1/{mode}-kl_loss', avg_kl_loss, global_step=global_step
                )

        if 'layers' in kwargs:
            for i, layer in enumerate(kwargs['layers']):
                # results.update({f"{key_prefix}": list(layer.adapterswitch.switch_logits.cpu())})
                for j, logit in enumerate(list(layer.adapterswitch.probs.cpu())):
                    results.update({f"layer_{i}/{mode}_{layer.used_adapter[j]}": logit.item()})
                results.update({f"tau": layer.adapterswitch.switch_temperature[0]})
        if 'norm_weights' in kwargs:
            for i, weight in enumerate(kwargs['norm_weights']):
                results.update({f"{mode}_norm_weights_{i}": weight})
        if 'lr' in kwargs:
            results.update({"lr": kwargs["lr"]})

        if 'f_lr' in kwargs:
            results.update({"f_lr": kwargs["f_lr"]})

        if "to_wandb" in kwargs and kwargs['to_wandb']:
            wandb.log(results, step=global_step)

        if mode in ["dev", "test"]:
            with open(Path(self.expdir) / f"{mode}_predict.txt", "w") as file:
                lines = [f"{f} {p}\n" for f, p in zip(records["filename"], records["predict_speaker"])]
                file.writelines(lines)

            with open(Path(self.expdir) / f"{mode}_truth.txt", "w") as file:
                lines = [f"{f} {l}\n" for f, l in zip(records["filename"], records["truth_speaker"])]
                file.writelines(lines)

        return save_names
