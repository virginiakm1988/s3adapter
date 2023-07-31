"""Downstream expert for Spoken Term Detection on Speech Commands."""

import re
import os
import hashlib
import sys
from pathlib import Path
from typing import List, Tuple, Union, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from catalyst.data.sampler import DistributedSamplerWrapper
from torch.distributed import is_initialized
from torch.nn.utils.rnn import pad_sequence

from ..model import *
from .dataset import SpeechCommandsDataset, SpeechCommandsTestingDataset, CLASSES

import wandb


class DownstreamExpert(nn.Module):
    """
    Used to handle downstream-specific operations
    eg. downstream forward, metric computation, contents to log
    """

    def __init__(self, upstream_dim: int, downstream_expert: dict, expdir: str, **kwargs):
        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.datarc = downstream_expert["datarc"]
        self.modelrc = downstream_expert["modelrc"]
        if 'adapterConfig' in kwargs:
            self.adapterConfig = kwargs['adapterConfig']
        else:
            self.adapterConfig = None
            print("[asr/expert.py] 105: No Adapter Config")

        train_list, valid_list = split_dataset(self.datarc["speech_commands_root"])
        num_paths = len(self.adapterConfig.adapter.switch.path)
        self.full_train_dataset = SpeechCommandsDataset(train_list, **self.datarc)
        # self.train_dataset, self.switch_train_dataset = torch.utils.data.random_split(full_train_dataset, [1 - switch_ratio, switch_ratio])
        # self.switch_train_dataset = SpeechCommandsDataset(train_list['eval'], **self.datarc)
        self.dev_dataset = SpeechCommandsDataset(valid_list, **self.datarc)
        self.test_dataset = SpeechCommandsTestingDataset(**self.datarc)

        model_cls = eval(self.modelrc['select'])
        model_conf = self.modelrc.get(self.modelrc['select'], {})
        self.projector = nn.Linear(upstream_dim, self.modelrc['projector_dim'])
        self.model = model_cls(
            input_dim = self.modelrc['projector_dim'],
            output_dim = self.full_train_dataset.class_num,
            **model_conf,
        )
        self.curr_projector = self.projector
        self.curr_model = self.model
        if 'do_virtual' in kwargs and kwargs['do_virtual']:
            self.virtual_projector = nn.Linear(upstream_dim, self.modelrc['projector_dim'])
            self.virtual_model = model_cls(
                input_dim = self.modelrc['projector_dim'],
                output_dim = self.full_train_dataset.class_num,
                **model_conf,
            )
            for virtual_p in self.virtual_projector.parameters():
                setattr(virtual_p, '__is_virtual__', True)
            for virtual_p in self.virtual_model.parameters():
                setattr(virtual_p, '__is_virtual__', True)

        self.objective = nn.CrossEntropyLoss()
        self.expdir = expdir
        self.register_buffer('best_score', torch.zeros(1))

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

    def _get_balanced_train_dataloader(self, dataset: torch.utils.data.Subset, drop_last=False):
        if len(dataset) == 0:
            return None
        
        sample_weights = [dataset.dataset.sample_weights[i] for i in dataset.indices]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        if is_initialized():
            sampler = DistributedSamplerWrapper(sampler)
        return DataLoader(
            dataset,
            sampler=sampler,
            batch_size=self.datarc["batch_size"],
            drop_last=drop_last,
            num_workers=self.datarc["num_workers"],
            collate_fn=dataset.dataset.collate_fn,
        )

    def _get_balanced_dev_dataloader(self, dataset, drop_last=False):
        return DataLoader(
            dataset,
            sampler=WeightedRandomSampler(
                dataset.sample_weights, len(dataset.sample_weights)
            ),
            batch_size=self.datarc["batch_size"],
            drop_last=drop_last,
            num_workers=self.datarc["num_workers"],
            collate_fn=dataset.collate_fn,
        )

    def _get_dataloader(self, dataset, **kwargs):
        return DataLoader(
            dataset,
            shuffle=False,
            batch_size=self.datarc["batch_size"],
            drop_last=False,
            num_workers=self.datarc["num_workers"],
            collate_fn=dataset.collate_fn,
        )

    # Interface
    def get_dataloader(self, mode, epoch=None, **kwargs):
        if mode == 'train':
            switch_ratio = self.adapterConfig.adapter.switch.ratio
            self.train_dataset, self.switch_train_dataset = torch.utils.data.random_split(self.full_train_dataset, [1 - switch_ratio, switch_ratio])  
            return {'train': self._get_balanced_train_dataloader(self.train_dataset, drop_last=True),
                    'switch': self._get_balanced_train_dataloader(self.switch_train_dataset, drop_last=True)}
        elif mode == 'dev':
            return self._get_balanced_dev_dataloader(self.dev_dataset, drop_last=False)
        elif mode == 'test':
            return self._get_dataloader(self.test_dataset)
        else:
            raise NotImplementedError

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
        records["loss"].append(loss.item())
        records["acc"] += (predicted_classid == labels).view(-1).cpu().float().tolist()

        records["filename"] += filenames
        records["predict"] += [CLASSES[idx] for idx in predicted_classid.cpu().tolist()]
        records["truth"] += [CLASSES[idx] for idx in labels.cpu().tolist()]

        return loss

    # interface
    def log_records(self, mode, records, logger, global_step, **kwargs):
        save_names = []
        wandb.define_metric("dev-acc", summary="max")
        wandb.define_metric("dev-loss", summary="min")
        wandb.define_metric("train-acc", summary="max")
        wandb.define_metric("train-loss", summary="min")
        wandb.define_metric("train-total_loss", summary="min")

        results = {}
        key_prefix = f"{mode}"
        if 'layers' in kwargs:
            for i, layer in enumerate(kwargs['layers']):
                # results.update({f"{key_prefix}": list(layer.adapterswitch.switch_logits.cpu())})
                for j, logit in enumerate(list(layer.adapterswitch.probs.cpu())):
                    results.update({f"layer_{i}/{mode}_{layer.used_adapter[j]}": logit.item()})
                results.update({f"tau": layer.adapterswitch.switch_temperature[0]})
        if 'norm_weights' in kwargs:
            for i, weight in enumerate(kwargs['norm_weights']):
                results.update({f"{key_prefix}_norm_weights_{i}": weight})
        if 'lr' in kwargs:
            results.update({"lr": kwargs["lr"]})

        for key in ["loss", "acc"]:
            values = records[key]
            average = sum(values) / len(values)
            logger.add_scalar(
                f'speech_commands/{mode}-{key}',
                average,
                global_step=global_step
            )
            results.update({f'{mode}-ks-{key}': average})
            with open(Path(self.expdir, "log.log"), 'a') as f:
                if key == 'acc':
                    print(f"{mode} {key}: {average}")
                    f.write(f'{mode} at step {global_step}: {average}\n')
                    if mode == 'dev' and average > self.best_score:
                        self.best_score = (torch.ones(1) * average).to(self.best_score.device)
                        f.write(f'New best on {mode} at step {global_step}: {average}\n')
                        save_names.append(f'{mode}-best.ckpt')

        if mode == 'train': 
            average_aux_loss = torch.FloatTensor(records['aux_loss']).mean().item() if len(records['aux_loss']) > 0 else 0
            logger.add_scalar(
                f'speech_commands/{mode}-aux_loss', average_aux_loss, global_step=global_step
            )
            results.update({f'{mode}-aux_loss': average_aux_loss})
            #print(f'aux_loss {average_aux_loss}')

            total_loss = results['loss'] + average_aux_loss
            logger.add_scalar(
                f'speech_commands/{mode}-total_loss', total_loss, global_step=global_step
            )
            results.update({f'{mode}-total_loss': total_loss})

        wandb.log(results, step=global_step)

        with open(Path(self.expdir) / f"{mode}_predict.txt", "w") as file:
            lines = [f"{f} {i}\n" for f, i in zip(records["filename"], records["predict"])]
            file.writelines(lines)

        with open(Path(self.expdir) / f"{mode}_truth.txt", "w") as file:
            lines = [f"{f} {i}\n" for f, i in zip(records["filename"], records["truth"])]
            file.writelines(lines)

        return save_names


def split_dataset(
    root_dir: Union[str, Path], max_uttr_per_class=2 ** 27 - 1
) -> Tuple[Dict[str, List[Tuple[str, str]]], List[Tuple[str, str]]]:
    """Split Speech Commands into 3 set.
    
    Args:
        root_dir: speech commands dataset root dir
        max_uttr_per_class: predefined value in the original paper
    
    Return:
        train_list: [(class_name, audio_path), ...]
        valid_list: as above
    """
    train_list, valid_list = [], []

    for entry in Path(root_dir).iterdir():
        if not entry.is_dir() or entry.name == "_background_noise_":
            continue

        for audio_path in entry.glob("*.wav"):
            speaker_hashed = re.sub(r"_nohash_.*$", "", audio_path.name)
            hashed_again = hashlib.sha1(speaker_hashed.encode("utf-8")).hexdigest()
            percentage_hash = (int(hashed_again, 16) % (max_uttr_per_class + 1)) * (
                100.0 / max_uttr_per_class
            )

            if percentage_hash < 10:
                valid_list.append((entry.name, audio_path))
            elif percentage_hash < 20:
                pass  # testing set is discarded
            else:
                train_list.append((entry.name, audio_path))
            # elif percentage_hash < 20 + switch_ratio:
            #     train_list['eval'].append((entry.name, audio_path))
            # else:
            #     train_list['train'].append((entry.name, audio_path))

    return train_list, valid_list
