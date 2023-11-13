import os
import math
import torch
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, DistributedSampler
from torch.distributed import is_initialized
from torch.nn.utils.rnn import pad_sequence

from ..model import *
from .model import *
from .dataset import IEMOCAPDataset, collate_fn

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

        DATA_ROOT = self.datarc['root']
        meta_data = self.datarc["meta_data"]

        self.fold = self.datarc.get('test_fold') or kwargs.get("downstream_variant")
        if self.fold is None:
            self.fold = "fold1"

        print(f"[Expert] - using the testing fold: \"{self.fold}\". Ps. Use -o config.downstream_expert.datarc.test_fold=fold2 to change test_fold in config.")

        train_path = os.path.join(
            meta_data, self.fold.replace('fold', 'Session'), 'train_meta_data.json')
        print(f'[Expert] - Training path: {train_path}')

        test_path = os.path.join(
            meta_data, self.fold.replace('fold', 'Session'), 'test_meta_data.json')
        print(f'[Expert] - Testing path: {test_path}')
        
        dataset = IEMOCAPDataset(DATA_ROOT, train_path, self.datarc['pre_load'])
        trainlen = int((1 - self.datarc['valid_ratio']) * len(dataset))
        lengths = [trainlen, len(dataset) - trainlen]
        
        torch.manual_seed(0)
        self.train_dataset, self.dev_dataset = random_split(dataset, lengths)

        self.test_dataset = IEMOCAPDataset(DATA_ROOT, test_path, self.datarc['pre_load'])

        model_cls = eval(self.modelrc['select'])
        model_conf = self.modelrc.get(self.modelrc['select'], {})
        self.projector = nn.Linear(upstream_dim, self.modelrc['projector_dim'])
        self.model = model_cls(
            input_dim = self.modelrc['projector_dim'],
            output_dim = dataset.class_num,
            **model_conf,
        )
        self.curr_projector = self.projector
        self.curr_model = self.model
        if 'do_virtual' in kwargs and kwargs['do_virtual']:
            self.virtual_projector = nn.Linear(upstream_dim, self.modelrc['projector_dim'])
            self.virtual_model = model_cls(
                input_dim = self.modelrc['projector_dim'],
                output_dim = dataset.class_num,
                **model_conf,
            )
            for virtual_p in self.virtual_projector.parameters():
                setattr(virtual_p, '__is_virtual__', True)
            for virtual_p in self.virtual_model.parameters():
                setattr(virtual_p, '__is_virtual__', True)
        self.objective = nn.CrossEntropyLoss()
        self.expdir = expdir
        self.register_buffer('best_score', torch.zeros(1))
        if 'adapterConfig' in kwargs:
            self.adapterConfig = kwargs['adapterConfig']
        else:
            self.adapterConfig = None
            print("[asr/expert.py] 105: No Adapter Config")

    def get_downstream_name(self):
        return self.fold.replace('fold', 'emotion')


    def _get_train_dataloader(self, dataset):
        sampler = DistributedSampler(dataset) if is_initialized() else None
        return DataLoader(
            dataset, batch_size=self.datarc['train_batch_size'],
            shuffle=(sampler is None), sampler=sampler,
            num_workers=self.datarc['num_workers'],
            collate_fn=collate_fn
        )

    def _get_eval_dataloader(self, dataset):
        return DataLoader(
            dataset, batch_size=self.datarc['eval_batch_size'],
            shuffle=False, num_workers=self.datarc['num_workers'],
            collate_fn=collate_fn
        )

    def get_train_dataloader(self):
        return self._get_train_dataloader(self.train_dataset)

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
    def get_dataloader(self, mode, epoch=None):
        if mode == 'train':
            switch_ratio = self.adapterConfig.adapter.switch.ratio
            train_dataset, switch_dataset = torch.utils.data.random_split(self.train_dataset, [1 - switch_ratio, switch_ratio])
            return {"train": self._get_train_dataloader(train_dataset), "switch": None if len(switch_dataset) == 0 else self._get_train_dataloader(switch_dataset)}
        return eval(f'self.get_{mode}_dataloader')()

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
        records['acc'] += (predicted_classid == labels).view(-1).cpu().float().tolist()
        records['loss'].append(loss.item())

        records["filename"] += filenames
        records["predict"] += [self.test_dataset.idx2emotion[idx] for idx in predicted_classid.cpu().tolist()]
        records["truth"] += [self.test_dataset.idx2emotion[idx] for idx in labels.cpu().tolist()]

        return loss

    # interface
    def log_records(self, mode, records, logger, global_step, **kwargs):
        wandb.define_metric("dev-acc", summary="max")
        wandb.define_metric("train-acc", summary="max")
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
        if 'f_lr' in kwargs:
            results.update({"f_lr": kwargs['f_lr']})
            
        save_names = []
        for key in ["acc", "loss"]:
            values = records[key]
            average = torch.FloatTensor(values).mean().item()
            logger.add_scalar(
                f'emotion-{self.fold}/{mode}-{key}',
                average,
                global_step=global_step
            )
            results.update({f'{mode}-{key}': average})
            with open(Path(self.expdir) / "log.log", 'a') as f:
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
                f'emotion-{self.fold}/{mode}-aux_loss', average_aux_loss, global_step=global_step
            )
            results.update({f'{mode}-aux_loss': average_aux_loss})

            total_loss = results[f'{mode}-loss'] + average_aux_loss
            logger.add_scalar(
                f'emotion-{self.fold}/{mode}-total_loss', total_loss, global_step=global_step
            )
            results.update({f'{mode}-total_loss': total_loss})

        if mode in ["dev", "test"]:
            with open(Path(self.expdir) / f"{mode}_{self.fold}_predict.txt", "w") as file:
                line = [f"{f} {e}\n" for f, e in zip(records["filename"], records["predict"])]
                file.writelines(line)

            with open(Path(self.expdir) / f"{mode}_{self.fold}_truth.txt", "w") as file:
                line = [f"{f} {e}\n" for f, e in zip(records["filename"], records["truth"])]
                file.writelines(line)
        wandb.log(results, step=global_step)
        return save_names
