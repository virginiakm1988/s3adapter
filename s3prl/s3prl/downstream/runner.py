import os
import sys
import math
import glob
import uuid
import shutil
import random
import tempfile
import importlib
from pathlib import Path
from copy import deepcopy
from itertools import zip_longest, islice

import torch
import torchaudio
import numpy as np
import wandb
from tqdm import tqdm
import gc
from tensorboardX import SummaryWriter
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import is_initialized, get_rank, get_world_size
from torch.distributed.elastic.multiprocessing.errors import record as ddprecod
from s3prl import hub
from s3prl.optimizers import get_optimizer
from s3prl.schedulers import get_scheduler
from s3prl.upstream.interfaces import Featurizer
from s3prl.utility.helper import is_leader_process, get_model_state, show, defaultdict

####add prompt optimizer (for setting different lr for prompt)#######
from s3prl.prompt_optimizers import  get_prompt_optimizer

from huggingface_hub import HfApi, HfFolder, Repository
# add yaml support
import yaml
import logging
import time
linelogger = logging.getLogger(__name__)
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT)
linelogger.setLevel(logging.INFO)
from ..upstream.adapterModels import dict2obj, MyLogger, is_baseline, find_module
# linelogger = MyLogger(linelogger)

SAMPLE_RATE = 16000

MODEL_CARD_MARKDOWN = """---
datasets:
- superb
tags:
- library:s3prl
- benchmark:superb
- type:model
---

# Fine-tuned s3prl model

Upstream Model: {upstream_model}

## Model description

[More information needed]

## Intended uses & limitations

[More information needed]

## How to use

[More information needed]

## Limitations and bias

[More information needed]

## Training data

[More information needed]

## Training procedure

[More information needed]

## Evaluation results

[More information needed]

"""


class ModelEntry:
    def __init__(self, model, name, trainable, interfaces):
        self.model = model
        self.name = name
        self.trainable = trainable
        self.interfaces = interfaces


class Runner():
    """
    Used to handle high-level concepts of a ML experiment
    eg. training loop, evaluation loop, upstream propagation, optimization, logging, checkpoint saving
    """
    def __init__(self, args, config):
        self.args = args
        show(self.args)
        self.config = config

        self.stage2_ckpt = torch.load(self.args.stage2_ckpt, map_location='cpu') if self.args.stage2_ckpt else {}
        self.init_ckpt = torch.load(self.args.init_ckpt, map_location='cpu') if self.args.init_ckpt else {}

        '''
        if isinstance(args.upstream_adapter_config, str):   # In evaluate mode (i.e. run with -e evaluate), this parameter will be dict.
            with open(args.upstream_adapter_config, 'r') as file:
                self.adapterDict = yaml.load(file, Loader=yaml.FullLoader)
        else:
            self.adapterDict = args.upstream_adapter_config
        '''
        
        self.adapterDict = {'adapter': config['adapter_config']}
        self.adapter_config = dict2obj(self.adapterDict)
        self.adapter_config.adapter.switch.baseline = is_baseline(self.adapter_config.adapter.switch.baseline)
        
        if self.args.mode == 'train':
            # train both stage1 and stage2
            self.stage1_steps = int(self.args.stage1_ratio * self.config['runner']['total_steps'] * (not self.adapter_config.adapter.switch.baseline) // 2)
            self.stage2_steps = self.config['runner']['total_steps'] - self.stage1_steps * 2
            logging.warning(f"{self.adapter_config.adapter.switch.baseline}, {self.stage1_steps}")
        elif self.args.mode == 'train_stage1':
            self.stage1_steps = self.config['runner']['total_steps']
            self.stage2_steps = 0
        elif self.args.mode == 'train_stage2':
            self.stage1_steps = 0
            self.stage2_steps = self.config['runner']['total_steps']
        else:
            self.stage1_steps = self.stage2_steps = 1

        self.config['runner']['total_steps'] = self.stage1_steps + self.stage2_steps

        self.adapter_config.adapter.switch.tau.steps = self.stage1_steps
        if self.init_ckpt.get('Step', 0):
            init_step = self.init_ckpt['Step']
            self.init_ckpt['Step'] = init_step // 2 if init_step <= self.stage1_steps * 2 else init_step - self.stage1_steps
        self.adapter_config.adapter.switch.tau.init_steps = self.init_ckpt.get('Step', 0)
        self.adapter_config.adapter.switch.stage = 1 + (self.adapter_config.adapter.switch.tau.init_steps >= self.stage1_steps)
        self.stage = self.adapter_config.adapter.switch.stage
        
        linelogger.info(f"{self.adapter_config.adapter.switch.tau.init_steps}")
        linelogger.info(f"{self.adapter_config.adapter.switch.tau.steps}, {self.adapter_config.adapter.switch.tau.stop_value}")
        
        self.upstream = self._get_upstream()
        print(self.upstream.model)
        self.featurizer = self._get_featurizer()
        self.downstream = self._get_downstream()
        self.all_entries = [self.upstream, self.featurizer, self.downstream]
        if is_leader_process():
            wandb.init(
                project=f'{self.args.upstream}-{self.args.downstream}', 
                mode="online" if self.args.online else "disabled", 
                name=f'{int(self.args.stage1_ratio * 100)}% search, lr {self.config["optimizer"]["lr"]}'
            )
            newArg = self.args
            newArg.config = self.config
            newArg.upstream_adapter_config = self.adapterDict
            wandb.config.update(newArg)

    def _load_weight(self, model, name):
        init_weight = self.init_ckpt.get(name) if not 'optimizer' in name else self.init_ckpt.get('Optimizer')
        
        if init_weight:
            show(f'[Runner] - Loading {name} weights from the previous experiment')
            if 'optimizer' in name:
                model.load_state_dict(init_weight[name])
            else:
                model.load_state_dict(init_weight)
        
        if name == "Upstream":
            if "prefix" in sys.argv[-1]:
                # load prompt weight
                prompt_weight = self.init_ckpt.get("prompt")
                if prompt_weight:
                    show(f'[Runner] - Loading {"Prompt"} weights from the previous experiment')
                    model_dict = model.state_dict()
                    model_dict.update(prompt_weight)
                    model.load_state_dict(model_dict)
            if self.args.adapter:
                adapter_weight = self.init_ckpt.get('adapter')
                if adapter_weight:
                    show(f'[Runner] - Loading {"Adapter"} weights & switch logits from the previous experiment')
                    model_dict = model.state_dict()
                    model_dict.update(adapter_weight)
                    model.load_state_dict(model_dict)
                else:
                    upstream_weight = self.init_ckpt.get('Upstream')
                    if upstream_weight:
                        show(f'[Runner] - Loading {"Adapter"} weights & switch logits from the stage1 experiment')
                        # init ckpt is given
                        assert self.stage2_ckpt == {}, \
                            "init_ckpt and stage2_ckpt shouldn't be provided simultaneously."
                        
                        model_dict = model.state_dict()
                        for para, value in upstream_weight.items():
                            if any(delta_module in para for delta_module in ['adapter', 'lora', 'lnfit', 'bitfit']):#'adapter' in para:
                                if 'switch' in para and self.adapter_config.adapter.switch.baseline:
                                    continue
                                model_dict.update({para: value})
                        model.load_state_dict(model_dict)
        elif name == 'Downstream':
            downstream_weight = self.init_ckpt.get('Downstream')
            if downstream_weight:
                show(f'[Runner] - Loading downstream weights from the previous experiment')
                model_dict = model.state_dict()
                for para, value in downstream_weight.items():
                    model_dict.update({para: value})
                    show(f'{para}: {value}')
                model.load_state_dict(model_dict)
                
    def _init_model(self, model, name, trainable, interfaces=None):
        for interface in interfaces or []:
            assert hasattr(model, interface), interface

        self._load_weight(model, name)

        if is_initialized() and trainable and any((p.requires_grad for p in model.parameters())):
            model = DDP(model, device_ids=[self.args.local_rank], find_unused_parameters=True)
            for interface in interfaces or []:
                setattr(model, interface, getattr(model.module, interface))

        return ModelEntry(model, name, trainable, interfaces)


    def _get_upstream(self):
        if "from_hf_hub" in self.args and self.args.from_hf_hub == True:
            from huggingface_hub import snapshot_download

            print(f'[Runner] - Downloading upstream model {self.args.upstream} from the Hugging Face Hub')
            filepath = snapshot_download(self.args.upstream, self.args.upstream_revision, use_auth_token=True)
            sys.path.append(filepath)

            dependencies = (Path(filepath) / 'requirements.txt').resolve()
            print("[Dependency] - The downloaded upstream model requires the following dependencies. Please make sure they are installed:")
            for idx, line in enumerate((Path(filepath) / "requirements.txt").open().readlines()):
                print(f"{idx}. {line.strip()}")
            print(f"You can install them by:")
            print()
            print(f"pip install -r {dependencies}")
            print()

            from expert import UpstreamExpert
            Upstream = UpstreamExpert
            ckpt_path = os.path.join(filepath, self.args.upstream_model_name)
        else:
            Upstream = getattr(hub, self.args.upstream)
            ckpt_path = self.args.upstream_ckpt
        upstream_refresh = self.args.upstream_refresh

        if is_initialized() and get_rank() > 0:
            torch.distributed.barrier()
            upstream_refresh = False

        model = Upstream(
            ckpt = ckpt_path,
            model_config = self.args.upstream_model_config,
            refresh = upstream_refresh,
            adapterConfig = self.adapter_config,
        ).to(self.args.device)

        if 'lnfit' in self.adapter_config.adapter.type:
            for name, _ in model.model.named_parameters():
                if '_layer_norm' in name and 'lnfit' not in name:
                    parent_key = ".".join(name.split('.')[:-1])
                    parent, _, original_ln = find_module(model.model, parent_key)
                    child_key =["deltaList", f"{parent.adapterIdx['lnfit']}", f"lnfit_{name.split('.')[-2]}"]
                    for key in child_key[:-1]:
                        parent = getattr(parent, key)
                    setattr(
                        parent, 
                        child_key[-1], 
                        deepcopy(original_ln)
                    )
                    # Remove bitfit_bias from lnfit's layer_norm
                    if 'bitfit' in self.adapter_config.adapter.type:
                        delattr(
                            getattr(parent, child_key[-1]),
                            'bitfit_bias'
                        )

        if is_initialized() and get_rank() == 0:
            torch.distributed.barrier()

        return self._init_model(
            model = model,
            name = 'Upstream',
            trainable = self.args.upstream_trainable,
            interfaces = ["get_downsample_rates"]
        )


    def _get_featurizer(self):
        model = Featurizer(
            upstream = self.upstream.model,
            feature_selection = self.args.upstream_feature_selection,
            layer_selection = self.args.upstream_layer_selection,
            upstream_device = self.args.device,
            normalize = self.args.upstream_feature_normalize,
        ).to(self.args.device)

        return self._init_model(
            model = model,
            name = 'Featurizer',
            trainable = True,
            interfaces = ['output_dim', 'downsample_rate']
        )


    def _get_downstream(self):
        expert = importlib.import_module(f"s3prl.downstream.{self.args.downstream}.expert")
        Downstream = getattr(expert, "DownstreamExpert")

        model = Downstream(
            upstream_dim = self.featurizer.model.output_dim,
            upstream_rate = self.featurizer.model.downsample_rate,
            **self.config,
            **vars(self.args),
            adapterConfig = self.adapter_config
        ).to(self.args.device)
        # model.adapterConfig = self.adapter_config

        return self._init_model(
            model = model,
            name = 'Downstream',
            trainable = True,
            interfaces = ['get_dataloader', 'log_records']
        )

    ### add prompt
    def _get_optimizer(self, model_params, name, prompt_lst=None):
        if "prefix" in sys.argv[-1]:
            optimizer = get_prompt_optimizer(
            model_params, 
            prompt_lst, ###
            self.config['runner']['total_steps'],
            self.config['optimizer']
            )
        else:
            optimizer = get_optimizer(
                model_params, 
                self.config['runner']['total_steps'],
                self.config['optimizer']
            )
        self._load_weight(optimizer, name)
        return optimizer


    def _get_scheduler(self, optimizer, total_steps=0, scheduler_name='Scheduler'):
        scheduler = get_scheduler(
            optimizer,
            self.config['runner']['total_steps'] if total_steps == 0 else total_steps,
            self.config['scheduler']
        )
        self._load_weight(scheduler, scheduler_name)
        return scheduler

    def _create_model_card(self, path):
        model_card = MODEL_CARD_MARKDOWN.format(upstream_model=self.args.upstream)
        with open(os.path.join(path, "README.md"), "w") as f:
            f.write(model_card)

    def fetch_dataloader(self, mode: str):
        return self.downstream.model.get_dataloader(mode)

    def prepare_stage(self, stage: int):
        if isinstance(self.upstream.model):
            self.upstream.model.module.model.set_stage(stage)
        else:
            self.upstream.model.model.set_stage(stage)

        # self.upstream.model.model.set_stage(stage)
        for entry in self.all_entries:
            if self.args.adapter != False and entry.name == "Upstream":
                for name, param in entry.model.named_parameters():
                    if any(delta_module in name for delta_module in ['adapter', 'lora', 'bitfit', 'lnfit']):#"adapter" in name or 'lora' in name or 'bitfit' in name or 'lnfit' in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
            if entry.name == "Featurizer":
                for name, param in entry.model.named_parameters():
                    param.requires_grad = (stage == 1 and self.args.stage1_weighted_sum) \
                                            or (stage == 2 and self.args.stage2_weighted_sum) \
                                                or (self.args.f_lr and stage >= self.args.f_lr_stage)
        if stage == 1:
            # if not self.args.stage1_weighted_sum:
            if not self.args.f_lr or self.args.f_lr_stage == 2:
                self.featurizer.model.eval()
            else:
                linelogger.info(f'train f_lr at stage 1')
                self.featurizer.model.train()
        elif stage == 2:
            if isinstance(self.downstream.model, DDP):
                self.downstream.model.module.adapterConfig.adapter.switch.ratio = 0
            else:
                self.downstream.model.adapterConfig.adapter.switch.ratio = 0
            self.featurizer.model.train()
    @ddprecod
    def train(self):
        # trainable parameters and train/eval mode
        trainable_paras = []
        # Network weights
        trainable_w_paras = []
        # Architecture weights (switch)
        trainable_a_paras = []
        # Featurizer paras
        trainable_f_paras = []

        additional_weight = [] # add prompt paras to optimizer
        for entry in self.all_entries:
            #### add the weight of prefix ###############
            if (self.args.prompt[0] == "prefix" or self.args.prompt[0] == "preinput") and entry.name == "Upstream":
                for  name, param in entry.model.named_parameters():
                    if "prompt" in name:
                        additional_weight.append(param)
                        param.requires_grad = True
                        print("Prompt!!",name)
                trainable_paras += list(additional_weight)

            if entry.trainable:
                entry.model.train()

            #### add adapters ##################
            if self.args.adapter != None and entry.name == "Upstream":
                adapter_param = 0
                for name, param in entry.model.named_parameters():
                    if any(delta_module in name for delta_module in ['adapter', 'lora', 'bitfit', 'lnfit']):#"adapter" in name or 'lora' in name or 'bitfit' in name or 'lnfit' in name:
                        param.requires_grad = True
                        if 'switch' in name:
                            trainable_a_paras.append(param)
                        else:
                            trainable_w_paras.append(param)
                            adapter_param += param.nelement() 
                    else:
                        param.requires_grad = False
                    
                trainable_paras += list(additional_weight)
                linelogger.info("Numbers of adapter PARAMETER: %.2fM" % (adapter_param/1e6))
            elif entry.name == 'Featurizer' and not self.adapter_config.adapter.switch.baseline and self.args.f_lr:
                linelogger.info("appending weight to f_optimizer")
                trainable_f_paras += list(entry.model.parameters())
            elif entry.trainable:
                linelogger.info(f"append weights: {entry.name}, {len(list(entry.model.parameters()))}")
                trainable_w_paras += list(entry.model.parameters())
            else:
                linelogger.info(f'in eval: {entry.name}')
                entry.model.eval()

        # optimizer
        w_optimizer = self._get_optimizer(trainable_w_paras, 'w_optimizer', [])
        a_optimizer = self._get_optimizer(trainable_a_paras, 'a_optimizer', [])
        f_optimizer = None
        if len(trainable_f_paras):
            f_optimizer = self._get_optimizer(trainable_f_paras, 'f_optimizer', [])
        
        # scheduler
        scheduler = None
        f_scheduler = None
        if self.config.get('scheduler'):
            scheduler = self._get_scheduler(w_optimizer)
            if len(trainable_f_paras):
                f_scheduler = self._get_scheduler(
                    f_optimizer, 
                    self.stage2_steps if self.args.f_lr_stage == 2 else self.config['runner']['total_steps'], 
                    scheduler_name='f_scheduler'
                )

        # specaug
        specaug = None
        if self.config.get('specaug'):
            linelogger.info(f'specaug is on!')
            from .specaug import SpecAug
            specaug = SpecAug(**self.config["specaug"])

        # progress bar
        tqdm_file = sys.stderr if is_leader_process() else open(os.devnull, 'w')
        pbar = tqdm(total=self.config['runner']['total_steps'], dynamic_ncols=True, desc='overall', file=tqdm_file)
        init_step = self.init_ckpt.get('Step')
        if init_step:
            pbar.n = init_step

        # Tensorboard logging
        if is_leader_process():
            logger = SummaryWriter(self.args.expdir)

        backward_steps = 0
        delta_step = 0
        batch_ids = []
        records = defaultdict(list)
        epoch = self.init_ckpt.get('Epoch', {'train': 0, 'switch': 0})
        train_split = self.config['runner'].get("train_dataloader", "train")

        linelogger.info(f'train stage for {self.stage1_steps} steps')
        adapterModes = ['train', 'switch'] if len(self.adapter_config.adapter.switch.path) > 1 else ['train']            
        
        # Log initial tau, switch logits & norm_weight to wandb
        if is_leader_process() and self.args.online:
            if isinstance(self.upstream.model, DDP):
                layers, norm_weights = self.upstream.model.module.model.encoder.layers, self.featurizer.model.module.norm_weights.detach()
            else:
                layers, norm_weights = self.upstream.model.model.encoder.layers, self.featurizer.model.norm_weights.detach()
            results = {}
            for i, layer in enumerate(layers):
                for j, logit in enumerate(list(layer.adapterswitch.probs.cpu())):
                    results.update({f"layer_{i}/{train_split}_{layer.used_adapter[j]}": logit.item()})
                results.update({f"tau": layer.adapterswitch.switch_temperature[0]})
            
            for i, weight in enumerate(norm_weights):
                results.update({f"{train_split}_norm_weights_{i}": weight})

            if scheduler:
                results.update({"lr": scheduler.get_last_lr()[0]})
            wandb.log(results, step=pbar.n)
            del results

        linelogger.info(f"gradient accumulate steps: {self.config['runner'].get('gradient_accumulate_steps')}")
        self.prepare_stage(self.stage)
        try:
            dataloaders = self.downstream.model.get_dataloader(train_split, epoch=epoch)
        except TypeError as e:
            if "unexpected keyword argument 'epoch'" in str(e):
                try:
                    dataloaders = self.downstream.model.get_dataloader(train_split)
                    for adapterMode in adapterModes:
                        if hasattr(dataloaders[adapterMode], "sampler") and isinstance(dataloaders[adapterMode].sampler, DistributedSampler):
                            dataloaders[adapterMode].sampler.set_epoch(epoch)
                except:
                    raise
            else:
                raise
        
        indices = {}
        for adapterMode in adapterModes:
            if dataloaders[adapterMode]:
                linelogger.info(f'dataset size of {adapterMode}: {len(dataloaders[adapterMode].dataset)}')
                linelogger.info(f'data loader size of {adapterMode}: {len(dataloaders[adapterMode])}')
                linelogger.info(f'dataset # indice of {adapterMode}: {len(dataloaders[adapterMode].dataset.indices)}')
        if dataloaders['switch']:
            linelogger.info(f'dataset overlap: {len(set(dataloaders["train"].dataset.indices) & set(dataloaders["switch"].dataset.indices))}')

        input_modes, cur_step, iters = {}, {}, {}
        for adapterMode in adapterModes:
            if dataloaders[adapterMode]:
                input_modes[adapterMode] = None
                cur_step[adapterMode] = 0
                iters[adapterMode] = iter(dataloaders[adapterMode])
            
        self.stage_steps_prefix = [self.stage1_steps, self.config['runner']['total_steps']]
        gradient_accumulate_steps = self.config['runner'].get('gradient_accumulate_steps')
        sentinel = (object(), object) # To check if iterator reach EOF

        while pbar.n < self.config['runner']['total_steps']:
            if self.stage == 1 and pbar.n >= self.stage_steps_prefix[self.stage - 1]:
                self.prepare_stage(2)
                self.stage = 2
                adapterModes = ['train']
                delta_step = self.stage1_steps * 2 - pbar.n 
                try:
                    dataloaders = self.downstream.model.get_dataloader(train_split, epoch=epoch['train'])
                except TypeError as e:
                    if "unexpected keyword argument 'epoch'" in str(e):
                        try:
                            dataloaders = self.downstream.model.get_dataloader(train_split)
                            for adapterMode in adapterModes:
                                if hasattr(dataloaders[adapterMode], "sampler") and isinstance(dataloaders[adapterMode].sampler, DistributedSampler):
                                    dataloaders[adapterMode].sampler.set_epoch(epoch[adapterMode])
                                iters[adapterMode] = iter(dataloaders[adapterMode])
                        except:
                            raise
                    else:
                        raise
            
            batch_id = 0
            inner_pbar = tqdm(len(dataloaders['train']), dynamic_ncols=True, desc=f'train_stage{self.stage}', file=tqdm_file)
            while pbar.n < self.stage_steps_prefix[self.stage - 1]:
                for adapterMode in adapterModes:
                    assert(not (adapterMode == 'switch' and self.stage == 2))

                    optimizer, lr_scheduler, trainable_paras = \
                        (w_optimizer, scheduler, trainable_w_paras) if adapterMode == 'train' else (a_optimizer, None, trainable_a_paras)
                    if trainable_f_paras and self.stage >= self.args.f_lr_stage and adapterMode == self.args.f_lr_mode:
                        trainable_paras = trainable_paras + trainable_f_paras
                    
                    if self.stage < 2:
                        for param in trainable_w_paras:
                            param.requires_grad = (adapterMode == 'train')
                        for param in trainable_a_paras:
                            param.requires_grad = (adapterMode == 'switch')
                        if f_optimizer and self.stage >= self.args.f_lr_stage:
                            for param in trainable_f_paras:
                                param.requires_grad = (adapterMode == self.args.f_lr_mode)
                    
                    for _ in range(gradient_accumulate_steps):
                        try:
                            (wavs, *others) = next(iters[adapterMode])
                            if adapterMode == 'train':
                                batch_id += 1
                        except StopIteration:
                            if adapterMode == 'train':
                                batch_id = 0
                            epoch[adapterMode] += 1
                            if hasattr(dataloaders[adapterMode], "sampler") and isinstance(dataloaders[adapterMode].sampler, DistributedSampler):
                                dataloaders[adapterMode].sampler.set_epoch(epoch[adapterMode])
                            
                            # Reopen the pbar
                            if adapterMode == 'train':
                                inner_pbar.close()
                                inner_pbar = tqdm(len(dataloaders['train']), dynamic_ncols=True, desc=f'train_stage{self.stage}', file=tqdm_file)

                            iters[adapterMode] = iter(dataloaders[adapterMode])
                            (wavs, *others) = next(iters[adapterMode])
                        
                        # Input Data
                        input_modes[adapterMode] = {'wavs': wavs, 'others': others, 'add_weight': []}
                        try:     
                            global_step = pbar.n * (1 + (self.stage == 1)) + (1 + (adapterMode == 'switch')) + delta_step * (self.stage == 2)         
                            wavs = [torch.FloatTensor(wav).to(self.args.device) for wav in input_modes[adapterMode]['wavs']]
                            # Forward
                            if self.upstream.trainable:
                                features = self.upstream.model(wavs)
                            else:
                                with torch.no_grad():
                                    features = self.upstream.model(wavs)

                            features = self.featurizer.model(wavs, features)
                            if specaug:
                                features, _ = specaug(features)
                            loss = self.downstream.model(
                                train_split,
                                features, *input_modes[adapterMode]['others'],
                                records = records,
                            )
                            # Fair-DARTS
                            if adapterMode == 'switch' and self.adapter_config.adapter.switch.fair_darts:
                                if isinstance(self.upstream.model, DDP):
                                    loss += self.upstream.model.module.model.aux_loss() * self.adapter_config.adapter.switch.aux_loss_ratio
                                else:
                                    loss += self.upstream.model.model.aux_loss() * self.adapter_config.adapter.switch.aux_loss_ratio
                            
                            batch_ids.append(batch_id * (3 - self.stage) + (adapterMode == 'switch'))

                            (loss / gradient_accumulate_steps).backward()
                            del loss, wavs, others, features

                        except RuntimeError as e:
                            if 'CUDA out of memory' in str(e):
                                print(f'[Runner] - CUDA out of memory at step {global_step}, mode {adapterMode}')
                                if is_initialized():
                                    raise
                                with torch.cuda.device(self.args.device):
                                    torch.cuda.empty_cache()
                                optimizer.zero_grad()
                                continue
                            else:
                                raise
                        
                        if adapterMode == 'train':
                            inner_pbar.update(1)
                    
                    # gradient clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        trainable_paras, self.config['runner']['gradient_clipping'])

                    # optimize
                    if math.isnan(grad_norm):
                        print(f'[Runner] - grad norm is NaN at step {global_step}, mode {adapterMode}')
                    else:
                        optimizer.step()
                        if f_optimizer and (self.stage == 2 or (self.args.f_lr_stage == 1 and adapterMode == self.args.f_lr_mode)):
                            # only use f_optimizer if (1) it is not none and (2) at stage 2 or (training with self.args.f_lr_mode at stage 1) 
                            f_optimizer.step()
                            f_optimizer.zero_grad()
                            
                    optimizer.zero_grad()
                    
                    # adjust learning rate
                    if lr_scheduler:
                        lr_scheduler.step()

                    if f_scheduler and (self.stage == 2 or (self.args.f_lr_stage == 1 and adapterMode == self.args.f_lr_mode)):
                        f_scheduler.step()

                if self.stage < 2:
                    if isinstance(self.upstream.model, DDP):
                        self.upstream.model.module.model.reduce_tau()
                    else:
                        self.upstream.model.model.reduce_tau()
                
                pbar.update(1)
                
                if not is_leader_process():
                    batch_ids = []
                    records = defaultdict(list)
                    continue

                # logging
                if global_step % self.config['runner']['log_step'] == 0:
                    if isinstance(self.upstream.model, DDP):
                        layers, norm_weights = self.upstream.model.module.model.encoder.layers, self.featurizer.model.module.norm_weights.detach()
                    else:
                        layers, norm_weights = self.upstream.model.model.encoder.layers, self.featurizer.model.norm_weights.detach()
                    self.downstream.model.log_records(
                        train_split,
                        records = records,
                        logger = logger,
                        global_step = global_step,
                        batch_ids = batch_ids,
                        total_batch_num = len(dataloaders['train']),
                        adapter_mode = adapterMode,
                        layers = layers,  # add module after first model
                        norm_weights = norm_weights,
                        lr = scheduler.get_last_lr()[0] if scheduler else 0,
                        f_lr = f_scheduler.get_last_lr()[0] if (f_scheduler and self.stage >= self.args.f_lr_stage) else 0,
                        to_wandb = self.args.online
                    )
                    batch_ids = []
                    records = defaultdict(list)

                # evaluation and save checkpoint
                save_names = []

                if global_step % self.config['runner']['eval_step'] == 0:
                    for split in self.config['runner']['eval_dataloaders']:
                        save_names += self.evaluate(split, logger, global_step)

                if global_step % self.config['runner']['save_step'] == 0:
                    def check_ckpt_num(directory):
                        max_keep = self.config['runner']['max_keep']
                        ckpt_pths = glob.glob(f'{directory}/states-*.ckpt')
                        if len(ckpt_pths) >= max_keep:
                            ckpt_pths = sorted(ckpt_pths, key=lambda pth: int(pth.split('-')[-1].split('.')[0]))
                            for ckpt_pth in ckpt_pths[:len(ckpt_pths) - max_keep + 1]:
                                os.remove(ckpt_pth)
                    check_ckpt_num(self.args.expdir)
                    save_names.append(f'states-{global_step}.ckpt')

                if len(save_names) > 0:
                    all_states = {
                        'Optimizer': {
                            "w_optimizer": w_optimizer.state_dict(), 
                            "a_optimizer": a_optimizer.state_dict(),
                            "f_optimizer": None if not f_optimizer else f_optimizer.state_dict() # change the order of if-else may cause error
                        },
                        'Step': global_step,
                        'Epoch': epoch,
                        'Args': self.args,
                        'Config': self.config,
                    }

                    for entry in self.all_entries:
                        if entry.trainable:
                            all_states[entry.name] = get_model_state(entry.model)

                        if (self.args.prompt[0] == "prefix" or self.args.prompt[0] == "preinput") and entry.name == "Upstream":
                            prompt_state = {}
                            for name, param in entry.model.named_parameters():
                                if "prompt" in name:
                                    prompt_state[name] = param
                            all_states["prompt"] = prompt_state
                        
                        if self.args.adapter and entry.name == "Upstream":
                            if isinstance(entry.model, DDP):
                                named_paras = entry.model.module.named_parameters()
                            else:
                                named_paras = entry.model.named_parameters()
                            adapter_state = {}
                            for name, param in named_paras:
                                if any(delta_module in name for delta_module in ['adapter', 'lora', 'bitfit', 'lnfit']):
                                    adapter_state[name] = param
                            all_states["adapter"] = adapter_state

                    if scheduler:
                        all_states['Scheduler'] = scheduler.state_dict()

                    if f_scheduler:
                        all_states['f_scheduler'] = f_scheduler.state_dict()

                    if is_initialized():
                        all_states['WorldSize'] = get_world_size()

                    save_paths = [os.path.join(self.args.expdir, name) for name in save_names]
                    tqdm.write(f'[Runner] - Save the checkpoint to:')
                    for i, path in enumerate(save_paths):
                        tqdm.write(f'{i + 1}. {path}')
                        torch.save(all_states, path)
                    del all_states

            '''
            linelogger.info(f'dataset size of train: {len(dataloaders["train"].dataset)}, {pbar.n}')            
            for batch_id, (wavs, *others) in enumerate(tqdm(dataloaders['train'], dynamic_ncols=True, desc=f'train_stage{self.stage}', file=tqdm_file)):
                if pbar.n >= self.stage_steps_prefix[self.stage - 1]:
                    break
                
                valid_wavs, valid_others = None, None
                if self.stage < 2:
                    try:
                        (valid_wavs, *valid_others) = next(iters['switch'])
                    except StopIteration:
                        iters['switch'] = iter(dataloaders['switch'])
                        (valid_wavs, *valid_others) = next(iters['switch'])
                    input_modes['switch'] = {'wavs': valid_wavs, 'others': valid_others, 'add_weight': []}
                
                input_modes['train'] = {'wavs': wavs, 'others': others, 'add_weight': []}
                
                for adapterMode in adapterModes:
                    if self.stage >= 2 and adapterMode != 'train':
                        continue
                    
                    optimizer, lr_scheduler, trainable_paras = \
                        (w_optimizer, scheduler, trainable_w_paras) if adapterMode == 'train' else (a_optimizer, None, trainable_a_paras)
                    assert(not (adapterMode != 'train' and self.stage == 2))

                    if self.stage < 2:
                        for param in trainable_w_paras:
                            param.requires_grad = (adapterMode == 'train')
                        for param in trainable_a_paras:
                            param.requires_grad = (adapterMode == 'switch')
                        if f_optimizer and self.stage >= self.args.f_lr_stage:
                            for param in trainable_f_paras:
                                param.requires_grad = (adapterMode == self.args.f_lr_mode)
                    try:     
                        global_step = pbar.n * (1 + (self.stage == 1)) + (1 + (adapterMode == 'switch')) + delta_step * (self.stage == 2)                  
                        wavs = [torch.FloatTensor(wav).to(self.args.device) for wav in input_modes[adapterMode]['wavs']]
                        # linelogger.info(f"!!!!!{adapterMode} = = {self.stage}, {(wavs[0]).shape}, {wavs[0].device}, {'parent' if is_leader_process() else 'child'}")                        
                        if self.upstream.trainable:
                            features = self.upstream.model(wavs)
                        else:
                            with torch.no_grad():
                                features = self.upstream.model(wavs)

                        features = self.featurizer.model(wavs, features)
                        if specaug:
                            features, _ = specaug(features)
                        loss = self.downstream.model(
                            train_split,
                            features, *input_modes[adapterMode]['others'],
                            records = records,
                        )
                        if adapterMode == 'switch':
                            if is_initialized():
                                loss += self.upstream.model.module.model.aux_loss()
                            else:
                                loss += self.upstream.model.model.aux_loss()
                        
                        batch_ids.append(batch_id * (3 - self.stage) + (adapterMode == 'switch'))

                        gradient_accumulate_steps = self.config['runner'].get('gradient_accumulate_steps')
                        (loss / gradient_accumulate_steps).backward()
                        del loss, wavs, features

                    except RuntimeError as e:
                        if 'CUDA out of memory' in str(e):
                            print(f'[Runner] - CUDA out of memory at step {global_step}, mode {adapterMode}')
                            if is_initialized():
                                raise
                            with torch.cuda.device(self.args.device):
                                torch.cuda.empty_cache()
                            optimizer.zero_grad()
                            continue
                        else:
                            raise
                    
                    if adapterMode == 'train':
                        # Only increment backward_steps in one of the adapterModes
                        backward_steps += 1
                    
                    # linelogger.info(f"{backward_steps}, {gradient_accumulate_steps}")
                    if backward_steps % gradient_accumulate_steps > 0:
                        continue

                    # gradient clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        trainable_paras, self.config['runner']['gradient_clipping'])

                    # optimize
                    if math.isnan(grad_norm):
                        print(f'[Runner] - grad norm is NaN at step {global_step}, mode {adapterMode}')
                    else:
                        optimizer.step()
                        # with torch.cuda.device(self.args.device):
                        
                        if f_optimizer and (self.stage == 2 or (self.args.f_lr_stage == 1 and adapterMode == self.args.f_lr_mode)):
                            # only use f_optimizer if (1) it is not none and (2) at stage 2 or (training with self.args.f_lr_mode at stage 1) 
                            f_optimizer.step()
                            f_optimizer.zero_grad()
                            
                    optimizer.zero_grad()
                    
                    # adjust learning rate
                    if lr_scheduler:
                        # linelogger.info(lr_scheduler.get_last_lr()[0])
                        lr_scheduler.step()

                    if f_scheduler and (self.stage == 2 or (self.args.f_lr_stage == 1 and adapterMode == self.args.f_lr_mode)):
                        f_scheduler.step()
                        
                
                if backward_steps % gradient_accumulate_steps > 0:
                    continue
                
                if self.stage < 2:
                    if is_initialized():
                        self.upstream.model.module.model.reduce_tau()
                    else:
                        self.upstream.model.model.reduce_tau()
                
                pbar.update(1)
                
                if not is_leader_process():
                    batch_ids = []
                    records = defaultdict(list)
                    continue

                # logging
                if global_step % self.config['runner']['log_step'] == 0:
                    if is_initialized():
                        layers, norm_weights = self.upstream.model.module.model.encoder.layers, self.featurizer.model.module.norm_weights.detach()
                    else:
                        layers, norm_weights = self.upstream.model.model.encoder.layers, self.featurizer.model.norm_weights.detach()
                    self.downstream.model.log_records(
                        train_split,
                        records = records,
                        logger = logger,
                        global_step = global_step,
                        batch_ids = batch_ids,
                        total_batch_num = len(dataloaders['train']),
                        adapter_mode = adapterMode,
                        layers = layers,  # add module after first model
                        norm_weights = norm_weights,
                        lr = scheduler.get_last_lr()[0] if scheduler else 0,
                        f_lr = f_scheduler.get_last_lr()[0] if (f_scheduler and self.stage >= self.args.f_lr_stage) else 0,
                        to_wandb = True
                    )
                    batch_ids = []
                    records = defaultdict(list)

                # evaluation and save checkpoint
                save_names = []

                if global_step % self.config['runner']['eval_step'] == 0:
                    for split in self.config['runner']['eval_dataloaders']:
                        save_names += self.evaluate(split, logger, global_step)

                if global_step % self.config['runner']['save_step'] == 0:
                    def check_ckpt_num(directory):
                        max_keep = self.config['runner']['max_keep']
                        ckpt_pths = glob.glob(f'{directory}/states-*.ckpt')
                        if len(ckpt_pths) >= max_keep:
                            ckpt_pths = sorted(ckpt_pths, key=lambda pth: int(pth.split('-')[-1].split('.')[0]))
                            for ckpt_pth in ckpt_pths[:len(ckpt_pths) - max_keep + 1]:
                                os.remove(ckpt_pth)
                    check_ckpt_num(self.args.expdir)
                    save_names.append(f'states-{global_step}.ckpt')

                if len(save_names) > 0:
                    all_states = {
                        'Optimizer': {
                            "w_optimizer": w_optimizer.state_dict(), 
                            "a_optimizer": a_optimizer.state_dict(),
                            "f_optimizer": None if not f_optimizer else f_optimizer.state_dict() # change the order of if-else may cause error
                        },
                        'Step': global_step,
                        'Epoch': epoch,
                        'Args': self.args,
                        'Config': self.config,
                    }

                    for entry in self.all_entries:
                        if entry.trainable:
                            all_states[entry.name] = get_model_state(entry.model)

                        if (self.args.prompt[0] == "prefix" or self.args.prompt[0] == "preinput") and entry.name == "Upstream": ###
                            prompt_state = {}
                            for name, param in entry.model.named_parameters():
                                if "prompt" in name:
                                    prompt_state[name] = param
                            all_states["prompt"] = prompt_state
                        if self.args.adapter and entry.name == "Upstream": ###
                            adapter_state = {}
                            for name, param in entry.model.named_parameters():
                                if any(delta_module in name for delta_module in ['adapter', 'lora', 'bitfit', 'lnfit']):
                                    adapter_state[name] = param
                            all_states["adapter"] = adapter_state

                    if scheduler:
                        all_states['Scheduler'] = scheduler.state_dict()

                    if f_scheduler:
                        all_states['f_scheduler'] = f_scheduler.state_dict()

                    if is_initialized():
                        all_states['WorldSize'] = get_world_size()

                    save_paths = [os.path.join(self.args.expdir, name) for name in save_names]
                    tqdm.write(f'[Runner] - Save the checkpoint to:')
                    for i, path in enumerate(save_paths):
                        tqdm.write(f'{i + 1}. {path}')
                        torch.save(all_states, path)
            '''
                    
        pbar.close()

        if self.args.push_to_hf_hub:
            self.push_to_huggingface_hub()
        if is_leader_process():
            logger.close()
            wandb.finish()


    def evaluate(self, split=None, logger=None, global_step=0):
        """evaluate function will always be called on a single process even during distributed training"""

        # When this member function is called directly by command line
        not_during_training = split is None and logger is None and global_step == 0
        if not_during_training:
            split = self.args.evaluate_split
            tempdir = tempfile.mkdtemp()
            logger = SummaryWriter(tempdir)

        # fix seed to guarantee the same evaluation protocol across steps 
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.args.seed)
            with torch.cuda.device(self.args.device):
                torch.cuda.empty_cache()

        # record original train/eval states and set all models to eval
        trainings = []
        for entry in self.all_entries:
            trainings.append(entry.model.training)
            entry.model.eval()

        # prepare data
        dataloader = self.downstream.model.get_dataloader(split, self.args.mode)
        evaluate_ratio = float(self.config["runner"].get("evaluate_ratio", 1))
        evaluate_steps = round(len(dataloader) * evaluate_ratio)

        batch_ids = []
        records = defaultdict(list)
        for batch_id, (wavs, *others) in enumerate(tqdm(dataloader, dynamic_ncols=True, desc=split, total=evaluate_steps)):
            if batch_id > evaluate_steps:
                break

            wavs = [torch.FloatTensor(wav).to(self.args.device) for wav in wavs]
            with torch.no_grad():
                features = self.upstream.model(wavs)
                features = self.featurizer.model(wavs, features)
                self.downstream.model(
                    split,
                    features, *others,
                    records = records,
                    batch_id = batch_id,
                )
                batch_ids.append(batch_id)
                del wavs, features
        # logging
        save_names = self.downstream.model.log_records(
            split,
            records = records,
            logger = logger,
            global_step = global_step,
            batch_ids = batch_ids,
            total_batch_num = len(dataloader),
            to_wandb = (self.args.mode != 'evaluate')
        )
        batch_ids = []
        records = defaultdict(list)

        # prepare back to training
        if torch.cuda.is_available():
            with torch.cuda.device(self.args.device):
                torch.cuda.empty_cache()

        for entry, training in zip(self.all_entries, trainings):
            if training:
                entry.model.train()

        if not_during_training:
            logger.close()
            shutil.rmtree(tempdir)
        linelogger.info(save_names)
        return [] if type(save_names) is not list else save_names

    def inference(self):
        filepath = Path(self.args.evaluate_split)
        assert filepath.is_file(), filepath
        filename = filepath.stem

        if hasattr(self.downstream.model, "load_audio"):
            wav = self.downstream.model.load_audio(filepath)
        else:
            wav, sr = torchaudio.load(str(filepath))
            assert sr == SAMPLE_RATE, sr
        wavs = [wav.view(-1).to(self.args.device)]

        for entry in self.all_entries:
            entry.model.eval()

        with torch.no_grad():
            features = self.upstream.model(wavs)
            features = self.featurizer.model(wavs, features)
            self.downstream.model.inference(features, [filename])

    def push_to_huggingface_hub(self):
        """Creates a downstream repository on the Hub and pushes training artifacts to it."""
        if self.args.hf_hub_org.lower() != "none":
            organization = self.args.hf_hub_org
        else:
            organization = os.environ.get("HF_USERNAME")
        huggingface_token = HfFolder.get_token()
        print(f"[Runner] - Organisation to push fine-tuned model to: {organization}")
        
        # Extract upstream repository metadata
        if self.args.hub == "huggingface":
            model_info = HfApi().model_info(self.args.upstream, token=huggingface_token)
            downstream_model_id = model_info.sha
            # Exclude "/" characters from downstream repo ID
            upstream_model_id = model_info.modelId.replace("/", "__")
        else:
            upstream_model_id = self.args.upstream.replace("/", "__")
            downstream_model_id = str(uuid.uuid4())[:8]
        repo_name = f"{upstream_model_id}__{downstream_model_id}"
        # Create downstream repo on the Hub
        repo_url = HfApi().create_repo(
            token=huggingface_token,
            name=repo_name,
            organization=organization,
            exist_ok=True,
            private=False,
        )
        print(f"[Runner] - Created Hub repo: {repo_url}")

        # Download repo
        HF_HUB_DIR = "hf_hub"
        REPO_ROOT_DIR = os.path.join(self.args.expdir, HF_HUB_DIR, repo_name)
        REPO_TASK_DIR = os.path.join(REPO_ROOT_DIR, self.args.downstream, self.args.expname)
        print(f"[Runner] - Cloning Hub repo to {REPO_ROOT_DIR}")
        model_repo = Repository(
            local_dir=REPO_ROOT_DIR, clone_from=repo_url, use_auth_token=huggingface_token
        )
        # Pull latest changes if they exist
        model_repo.git_pull()

        # Copy checkpoints, tensorboard logs, and args / configs
        # Note that this copies all files from the experiment directory,
        # including those from multiple runs
        shutil.copytree(self.args.expdir, REPO_TASK_DIR, dirs_exist_ok=True, ignore=shutil.ignore_patterns(HF_HUB_DIR))

        # By default we use model.ckpt in the PreTrainedModel interface, so
        # rename the best checkpoint to match this convention
        checkpoints = list(Path(REPO_TASK_DIR).glob("*best*.ckpt"))
        if len(checkpoints) == 0:
            print("[Runner] - Did not find a best checkpoint! Using the final checkpoint instead ...")
            CKPT_PATH = (
                os.path.join(REPO_TASK_DIR, f"states-{self.config['runner']['total_steps']}.ckpt")
                )
        elif len(checkpoints) > 1:
            print(f"[Runner] - More than one best checkpoint found! Using {checkpoints[0]} as default ...")
            CKPT_PATH = checkpoints[0]
        else:
            print(f"[Runner] - Found best checkpoint {checkpoints[0]}!")
            CKPT_PATH = checkpoints[0]
        shutil.move(CKPT_PATH, os.path.join(REPO_TASK_DIR, "model.ckpt"))
        model_repo.lfs_track("*.ckpt")

        # Write model card
        self._create_model_card(REPO_ROOT_DIR)

        # Push everything to the Hub
        print("[Runner] - Pushing model files to the Hub ...")
        model_repo.push_to_hub()
        print("[Runner] - Training run complete!")
