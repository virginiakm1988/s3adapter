import os
import math
import torch
import random
import pandas as pd
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed import is_initialized
from torch.nn.utils.rnn import pad_sequence

from ..model import *
from .dataset import FluentCommandsDataset

import wandb

class DownstreamExpert(nn.Module):
    """
    Used to handle downstream-specific operations
    eg. downstream forward, metric computation, contents to log
    """

    def __init__(self, upstream_dim, downstream_expert, expdir, **kwargs):
        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.datarc = downstream_expert['datarc']
        self.modelrc = downstream_expert['modelrc']
        if 'adapterConfig' in kwargs:
            self.adapterConfig = kwargs['adapterConfig']
            switch_ratio = self.adapterConfig.adapter.switch.ratio
        else:
            self.adapterConfig = None
            print("[asr/expert.py] 105: No Adapter Config")

        self.get_dataset()

        
        self.full_train_dataset = FluentCommandsDataset(self.train_df, self.base_path, self.Sy_intent)
        # self.switch_train_dataset = FluentCommandsDataset(self.switch_train_df, self.base_path, self.Sy_intent)
        self.dev_dataset = FluentCommandsDataset(self.valid_df, self.base_path, self.Sy_intent)
        self.test_dataset = FluentCommandsDataset(self.test_df, self.base_path, self.Sy_intent)

        model_cls = eval(self.modelrc['select'])
        model_conf = self.modelrc.get(self.modelrc['select'], {})
        self.projector = nn.Linear(upstream_dim, self.modelrc['projector_dim'])
        self.model = model_cls(
            input_dim = self.modelrc['projector_dim'],
            output_dim = sum(self.values_per_slot),
            **model_conf,
        )
        self.curr_projector = self.projector
        self.curr_model = self.model
        if 'do_virtual' in kwargs and kwargs['do_virtual']:
            self.virtual_projector = nn.Linear(upstream_dim, self.modelrc['projector_dim'])
            self.virtual_model = model_cls(
                input_dim = self.modelrc['projector_dim'],
                output_dim = sum(self.values_per_slot),
                **model_conf,
            )
            for virtual_p in self.virtual_projector.parameters():
                setattr(virtual_p, '__is_virtual__', True)
            for virtual_p in self.virtual_model.parameters():
                setattr(virtual_p, '__is_virtual__', True)
        self.objective = nn.CrossEntropyLoss()
        self.expdir = expdir
        self.register_buffer('best_score', torch.zeros(1))

    def get_dataset(self):
        # ToDo: split training dataset 
        self.base_path = self.datarc['file_path']
        train_df = pd.read_csv(os.path.join(self.base_path, "data", "train_data.csv"))
        valid_df = pd.read_csv(os.path.join(self.base_path, "data", "valid_data.csv"))
        test_df = pd.read_csv(os.path.join(self.base_path, "data", "test_data.csv"))

        Sy_intent = {"action": {}, "object": {}, "location": {}}

        values_per_slot = []
        for slot in ["action", "object", "location"]:
            slot_values = Counter(train_df[slot])
            for index, value in enumerate(slot_values):
                Sy_intent[slot][value] = index
                Sy_intent[slot][index] = value
            values_per_slot.append(len(slot_values))
        self.values_per_slot = values_per_slot
        self.Sy_intent = Sy_intent
        # self.switch_train_df = train_df.sample(frac=self.datarc['switch_ratio'])
        # self.train_df = train_df.drop(self.switch_train_df.index).reset_index()
        # self.switch_train_df = self.switch_train_df.reset_index()
        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df

    def _get_train_dataloader(self, dataset):
        if len(dataset) == 0:
            return None
        
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

    def get_switch_dataloader(self):
        return self._get_train_dataloader(self.switch_train_dataset)

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
    def get_dataloader(self, mode, epoch=None, **kwargs):
        if mode == 'train':
            switch_ratio = self.adapterConfig.adapter.switch.ratio
            self.train_dataset, self.switch_train_dataset = \
                torch.utils.data.random_split(self.full_train_dataset, [1 - switch_ratio, switch_ratio])
            return {'train': eval(f'self.get_train_dataloader')(),
                    'switch': eval(f'self.get_switch_dataloader')()}
        else:
            return eval(f'self.get_{mode}_dataloader')()

    # Interface
    def forward(self, mode, features, labels, filenames, records, **kwargs):
        labels = [torch.LongTensor(label) for label in labels]
        features_len = torch.IntTensor([len(feat) for feat in features]).to(device=features[0].device)
        features = pad_sequence(features, batch_first=True)

        features = self.curr_projector(features)
        intent_logits, _ = self.curr_model(features, features_len)

        intent_loss = 0
        start_index = 0
        predicted_intent = []
        
        labels = torch.stack(labels).to(features.device)
        for slot in range(len(self.values_per_slot)):
            end_index = start_index + self.values_per_slot[slot]
            subset = intent_logits[:, start_index:end_index]

            intent_loss += self.objective(subset, labels[:, slot])
            predicted_intent.append(subset.max(1)[1])
            start_index = end_index

        predicted_intent = torch.stack(predicted_intent, dim=1)
        records['acc'] += (predicted_intent == labels).prod(1).view(-1).cpu().float().tolist()
        records['intent_loss'].append(intent_loss.item())

        def idx2slots(indices: torch.Tensor):
            action_idx, object_idx, location_idx = indices.cpu().tolist()
            return (
                self.Sy_intent["action"][action_idx],
                self.Sy_intent["object"][object_idx],
                self.Sy_intent["location"][location_idx],
            )

        records["filename"] += filenames
        records["predict"] += list(map(idx2slots, predicted_intent))
        records["truth"] += list(map(idx2slots, labels))

        return intent_loss

    # interface
    def log_records(self, mode, records, logger, global_step, **kwargs):
        save_names = []
        wandb.define_metric("test-acc", summary="max")
        wandb.define_metric("test-intent_loss", summary="min")
        wandb.define_metric("dev-acc", summary="max")
        wandb.define_metric("dev-intent_loss", summary="min")
        wandb.define_metric("train-acc", summary="max")
        wandb.define_metric("train-intent_loss", summary="min")
        wandb.define_metric("train-total_loss", summary="min")

        results = {}
        key_prefix = f"{mode}"
        if 'layers' in kwargs:
            for i, layer in enumerate(kwargs['layers']):
                # results.update({f"{key_prefix}": list(layer.adapterswitch.switch_logits.cpu())})
                for j, logit in enumerate(list(layer.adapterswitch.probs.cpu())):
                    results.update({f"layer_{i}/{key_prefix}_{layer.used_adapter[j]}": logit.item()})
                results.update({f"tau": layer.adapterswitch.switch_temperature[0]})
        if 'norm_weights' in kwargs:
            for i, weight in enumerate(kwargs['norm_weights']):
                results.update({f"{key_prefix}_norm_weights_{i}": weight})
        if 'lr' in kwargs:
            results.update({"lr": kwargs["lr"]})


        for key in ["acc", "intent_loss"]:
            values = records[key]
            average = torch.FloatTensor(values).mean().item()
            logger.add_scalar(
                f'fluent_commands/{mode}-{key}',
                average,
                global_step=global_step
            )
            results.update({key: average})
            with open(Path(self.expdir) / "log.log", 'a') as f:
                if key == 'acc':
                    print(f"{mode} {key}: {average}")
                    f.write(f'{mode} at step {global_step}: {average}\n')
                    if mode == 'dev' and average > self.best_score:
                        self.best_score = torch.ones(1).to(self.best_score.device) * average
                        f.write(f'New best on {mode} at step {global_step}: {average}\n')
                        save_names.append(f'{mode}-best.ckpt')

        if mode == 'train': 
            average_aux_loss = torch.FloatTensor(records['aux_loss']).mean().item() if len(records['aux_loss']) > 0 else 0
            logger.add_scalar(
                f'fluent_commands/{mode}-aux_loss', average_aux_loss, global_step=global_step
            )
            results.update({f'{mode}-aux_loss': average_aux_loss})
            #print(f'aux_loss {average_aux_loss}')

            total_loss = torch.FloatTensor(records['intent_loss']).mean().item() + average_aux_loss
            logger.add_scalar(
                f'fluent_commands/{mode}-total_loss', total_loss, global_step=global_step
            )
            results.update({f'{mode}-total_loss': total_loss})

        wandb.log(results, step=global_step)

        with open(Path(self.expdir) / f"{mode}_predict.csv", "w") as file:
            lines = [f"{f},{a},{o},{l}\n" for f, (a, o, l) in zip(records["filename"], records["predict"])]
            file.writelines(lines)

        with open(Path(self.expdir) / f"{mode}_truth.csv", "w") as file:
            lines = [f"{f},{a},{o},{l}\n" for f, (a, o, l) in zip(records["filename"], records["truth"])]
            file.writelines(lines)

        return save_names
