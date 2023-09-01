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
import torch.optim._functional as optim_F
import json

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
        
        self.prepare_baseline()

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
        
        self.do_virtual = not self.adapter_config.adapter.switch.baseline and \
            self.stage == 1 and self.adapter_config.adapter.switch.algo.name in ['darts', 'fair_darts', 'gumbel_darts', 's3delta'] \
                and (self.adapter_config.adapter.switch.algo.first_order or self.adapter_config.adapter.switch.algo.second_order)
        
        linelogger.info(f'do_virtual: {self.do_virtual}, stage = {self.stage}, switch.stage = {self.adapter_config.adapter.switch.stage}')
        if self.do_virtual:
            assert not (self.adapter_config.adapter.switch.algo.second_order and not self.adapter_config.adapter.switch.algo.first_order),\
            "Second order should be calculated when first order is enable."

        self.upstream = self._get_upstream()
        self.featurizer = self._get_featurizer()
        self.downstream = self._get_downstream()
        self.all_entries = [self.upstream, self.featurizer, self.downstream]

        # init wandb
        if is_leader_process():
            wandb_name = f'{self.args.search_algo}, {int(self.args.stage1_ratio * 100)}% search, lr {self.config["optimizer"]["lr"]}' if not self.adapter_config.adapter.switch.baseline else f'{self.args.search_algo} retrain'
            if self.args.random_exp:
                wandb_name = f'random exp {self.args.rand_seq}, budget={self.adapter_config.adapter.switch.algo.para_budget}, lr_rescheduled, same initialization'
            wandb.init(
                project=f'{self.args.upstream}-{self.args.downstream}', 
                mode="online" if self.args.online else "disabled", 
                name=wandb_name
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
                model_dict = model.state_dict()
                model_dict.update(init_weight)
                model.load_state_dict(model_dict)
        
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
        
        if self.args.random_exp:
            init_weight = torch.load(f'random_exp/{self.args.downstream}/init_weights.ckpt')
            if name == 'Upstream':
                show(f'[Runner] - Loading {name} initialize weights')
                for layer in model.get_layers:
                    for adapter in layer.delta_list:
                        adapter_dict = adapter.state_dict()
                        init_dict = deepcopy(init_weight['adapters'][adapter.name])
                        adapter_dict.update(init_dict)
                        adapter.load_state_dict(adapter_dict)
            elif name == 'Downstream':
                show(f'[Runner] - Loading {name} initialize weights')
                downstream_dict = model.state_dict()
                downstream_dict.update(deepcopy(init_weight['downstream']))
                model.load_state_dict(downstream_dict)

                
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
            interfaces = [
                "get_downsample_rates", 
                "reduce_tau", 
                "sample_gumbel", 
                "sample_uniform",
                "compute_shifted_sigmoid",
                "set_hard_forward_structure",
                "aux_loss", 
                "set_stage", 
                "use_virtual", 
                "use_default",
                "get_layers",
                "get_named_parameters"
            ]
        )


    def _get_featurizer(self):
        model = Featurizer(
            upstream = self.upstream.model,
            feature_selection = self.args.upstream_feature_selection,
            layer_selection = self.args.upstream_layer_selection,
            upstream_device = self.args.device,
            normalize = self.args.upstream_feature_normalize,
            do_virtual = self.do_virtual
        ).to(self.args.device)

        return self._init_model(
            model = model,
            name = 'Featurizer',
            trainable = True,
            interfaces = ['output_dim', 'downsample_rate', 'use_virtual', 'use_default', 'get_norm_weights']
        )


    def _get_downstream(self):
        expert = importlib.import_module(f"s3prl.downstream.{self.args.downstream}.expert")
        Downstream = getattr(expert, "DownstreamExpert")

        model = Downstream(
            upstream_dim = self.featurizer.model.output_dim,
            upstream_rate = self.featurizer.model.downsample_rate,
            **self.config,
            **vars(self.args),
            adapterConfig = self.adapter_config,
            do_virtual = self.do_virtual
        ).to(self.args.device)

        return self._init_model(
            model = model,
            name = 'Downstream',
            trainable = True,
            interfaces = ['get_dataloader', 'log_records', 'use_virtual', 'use_default']
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
    
    def prepare_baseline(self):
        if self.args.random_exp:
            with open(self.args.rand_arch) as arch_f:
                all_arch = json.load(arch_f)
                arch = all_arch[str(self.args.rand_seq)]
                baseline = [[] for _ in range(12)]
                for layer_id, used_adapters in enumerate(arch):
                    baseline[int(layer_id)] = [
                        idx for idx, adapter_name in enumerate(self.adapter_config.adapter.type) 
                        if adapter_name in used_adapters
                    ]
        elif self.args.mode == 'evaluate' and self.adapter_config.adapter.switch.algo.name == 's3delta':
            assert self.init_ckpt, 'Should provide a checkpoint for evaluation'
            arch_path = os.path.join(os.path.dirname(self.args.init_ckpt), 'architecture.json')
            if os.path.exists(arch_path):
                baseline = [[] for _ in range(12)]
                with open(arch_path, 'r') as arch_f:
                    arch = json.load(arch_f)
                    for layer_id, used_adapters in arch.items():
                        baseline[int(layer_id)] = [
                            idx for idx, adapter_name in enumerate(self.adapter_config.adapter.type) 
                            if adapter_name in used_adapters
                        ]
        else:
            baseline = self.adapter_config.adapter.switch.baseline
        
        self.adapter_config.adapter.switch.baseline = is_baseline(baseline)

    def fetch_dataloader(self, mode: str):
        return self.downstream.model.get_dataloader(mode)

    def prepare_dataloader(self, train_split, adapterModes, epoch):
        try:
            dataloaders = self.downstream.model.get_dataloader(train_split, epoch=epoch)
        except TypeError as e:
            if "unexpected keyword argument 'epoch'" in str(e):
                try:
                    dataloaders = self.downstream.model.get_dataloader(train_split)
                    for adapterMode in adapterModes:
                        if hasattr(dataloaders[adapterMode], "sampler") and isinstance(dataloaders[adapterMode].sampler, DistributedSampler):
                            dataloaders[adapterMode].sampler.set_epoch(epoch[adapterMode])
                except:
                    raise
            else:
                raise
        # Log dataset info
        for adapterMode in adapterModes:
            if dataloaders[adapterMode]:
                linelogger.info(f'dataset size of {adapterMode}: {len(dataloaders[adapterMode].dataset)}')
                linelogger.info(f'data loader size of {adapterMode}: {len(dataloaders[adapterMode])}')
                linelogger.info(f'dataset # indice of {adapterMode}: {len(dataloaders[adapterMode].dataset.indices)}')
        
        if dataloaders['switch']:
            linelogger.info(f'dataset overlap: {len(set(dataloaders["train"].dataset.indices) & set(dataloaders["switch"].dataset.indices))}')

        iters = {adapterMode: iter(dataloaders[adapterMode]) for adapterMode in adapterModes}
        
        return dataloaders, iters

    def prepare_stage(self, stage: int):
        self.upstream.model.set_stage(stage)
        for entry in self.all_entries:
            if self.args.adapter != False and entry.name == "Upstream":
                for name, param in entry.model.named_parameters():
                    if getattr(param, '__is_delta__', False):#"adapter" in name or 'lora' in name or 'bitfit' in name or 'lnfit' in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
            if entry.name == "Featurizer":
                for name, param in entry.model.named_parameters():
                    param.requires_grad = (self.args.f_lr and stage >= self.args.f_lr_stage)
        if stage == 1:
            if not self.args.f_lr or self.args.f_lr_stage == 2:
                self.featurizer.model.eval()
            else:
                linelogger.info(f'train f_lr at stage 1')
                self.featurizer.model.train()
        elif stage == 2:
            if self.adapter_config.adapter.switch.algo.name == 's3delta':
                self.upstream.model.set_hard_forward_structure(max_num_param=self.adapter_config.adapter.switch.algo.para_budget, baseline=self.adapter_config.adapter.switch.baseline)
            if isinstance(self.downstream.model, DDP):
                self.downstream.model.module.adapterConfig.adapter.switch.ratio = 0
            else:
                self.downstream.model.adapterConfig.adapter.switch.ratio = 0
            self.upstream.model.train()
            self.featurizer.model.train()
    
    def gen_weight(self):
        """
        Generate weights for adpater modules and downstream model, 
        only used for the initialization consistency in random_exp
        """
        weights = {
            'adapters': {},
            'downstream': {}
        }
        for adapter in self.upstream.model.get_layers[0].delta_list:
            adapter_dict = adapter.state_dict()
            if adapter.name == 'lora':
                # print(adapter_dict)
                keys = [key for key in adapter_dict.keys() if 'lora' not in key]
                for key in keys:
                    del adapter_dict[key]
            
            weights['adapters'][adapter.name] = adapter_dict
            
        weights['downstream'] = self.downstream.model.state_dict()
        if not os.path.exists(f'random_exp/{self.args.downstream}'):
            os.mkdir(f'random_exp/{self.args.downstream}')
        torch.save(weights, f'random_exp/{self.args.downstream}/init_weights.ckpt')

    @ddprecod
    def train(self):
        # trainable parameters and train/eval mode
        trainable_paras = []
        # Network weights
        trainable_w_paras = []
        # Architecture weights (switch)
        trainable_a_paras = []
        # Virtual paras (for DARTS-based algorithm that are not using first-order approximation)
        trainable_v_paras = []

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
                    if getattr(param, '__is_delta__', False):
                        trainable_w_paras.append(param)
                        linelogger.info(f'add {name} into trainable_w_paras')
                        param.requires_grad = True
                        adapter_param += param.nelement()
                    elif getattr(param, '__is_virtual__', False):
                        trainable_v_paras.append(param)
                        linelogger.info(f'add {name} into trainable_v_paras')
                        param.requires_grad = True
                    elif getattr(param, '__is_switch__', False):
                        trainable_a_paras.append(param)
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                trainable_paras += list(additional_weight)
                linelogger.info("Numbers of adapter PARAMETER: %.2fM" % (adapter_param/1e6))
                wandb.config.update({'num_trainable_parameters': adapter_param/1e6})
            elif entry.trainable:
                for name, param in entry.model.named_parameters():
                    if getattr(param, '__is_virtual__', False):
                        trainable_v_paras.append(param)
                    else:
                        trainable_w_paras.append(param)
            else:
                linelogger.info(f'in eval: {entry.name}')
                entry.model.eval()

        # optimizer
        w_optimizer = self._get_optimizer(trainable_w_paras, 'w_optimizer', [])
        a_optimizer = self._get_optimizer(trainable_a_paras, 'a_optimizer', [])
        v_optimizer = self._get_optimizer(trainable_v_paras, 'v_optimizer', []) if len(trainable_v_paras) else None

        if v_optimizer:
            assert len(trainable_w_paras) == len(trainable_v_paras), "Number of trainable_w_paras should be equal to number of trainable_v_paras"
        # Make switch_logit's grad equals to zero tensor, rather than None
        with torch.no_grad():
            for p in trainable_a_paras:
                p.grad = torch.zeros_like(p.data)
            for p in trainable_w_paras:
                p.grad = torch.zeros_like(p.data)
            for p in trainable_v_paras:
                p.grad = torch.zeros_like(p.data)
        
        # scheduler
        scheduler = None
        if self.config.get('scheduler'):
            scheduler = self._get_scheduler(w_optimizer)

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

        delta_step = 0
        batch_ids = []
        records = defaultdict(list)
        epoch = self.init_ckpt.get('Epoch', {'train': 0, 'switch': 0})
        train_split = self.config['runner'].get("train_dataloader", "train")

        linelogger.info(f'train stage 1 for {self.stage1_steps} steps')
        if not self.adapter_config.adapter.switch.baseline and self.stage == 1:    # Do search
            adapterModes = ['train', 'switch'] if self.adapter_config.adapter.switch.algo.name in ['gdas'] else ['switch', 'train']
        else:
            adapterModes = ['train'] 
        
        # Log initial tau, switch logits & norm_weight to wandb
        if is_leader_process() and self.args.online:
            layers, norm_weights = self.upstream.model.get_layers, self.featurizer.model.get_norm_weights
            self.downstream.model.log_records(
                train_split,
                records = records,
                logger = logger,
                global_step = pbar.n,
                batch_ids = batch_ids,
                total_batch_num = 0,
                layers = layers,
                norm_weights = norm_weights,
                lr = scheduler.get_last_lr()[0] if scheduler else 0,
                to_wandb = self.args.online
            )

        self.prepare_stage(self.stage)
        dataloaders, iters = self.prepare_dataloader(train_split, adapterModes, epoch)

        gradient_accumulate_steps = self.config['runner'].get('gradient_accumulate_steps')
        self.stage_steps_prefix = [self.stage1_steps, self.config['runner']['total_steps']]
        if self.args.random_exp:
            self.stage_steps_prefix = [0, math.ceil(len(dataloaders['train'])/gradient_accumulate_steps)]
            self.config['runner']['total_steps'] = self.stage_steps_prefix[-1]
        
        while pbar.n < self.config['runner']['total_steps']:
            if self.stage == 1 and pbar.n >= self.stage_steps_prefix[self.stage - 1]:
                self.prepare_stage(2)
                self.stage = 2
                adapterModes = ['train']
                delta_step = self.stage1_steps * 2 - pbar.n 
                self.inner_pbar.close()
                dataloaders, iters = self.prepare_dataloader(train_split, adapterModes, epoch)
                
            batch_id = -1 # set to -1 so that the first batch's batch_id will be zero.
            self.inner_pbar = tqdm(total=len(dataloaders['train']), dynamic_ncols=True, desc=f'train_stage{self.stage}', file=tqdm_file)
            while pbar.n < self.stage_steps_prefix[self.stage - 1]:
                # Collect data
                all_data = defaultdict(list)
                for adapterMode in adapterModes:
                    for i in range(gradient_accumulate_steps):
                        try:
                            (wavs, *others) = next(iters[adapterMode])
                        except StopIteration:
                            epoch[adapterMode] += 1
                            if hasattr(dataloaders[adapterMode], "sampler") and isinstance(dataloaders[adapterMode].sampler, DistributedSampler):
                                dataloaders[adapterMode].sampler.set_epoch(epoch[adapterMode])
                            
                            iters[adapterMode] = iter(dataloaders[adapterMode])
                            (wavs, *others) = next(iters[adapterMode])
                        all_data[adapterMode].append({'wavs': wavs, 'others': others})
                # start training
                for mode_id, adapterMode in enumerate(adapterModes):
                    assert(not (adapterMode == 'switch' and self.stage == 2))
                    
                    if self.stage < 2:
                        for param in trainable_w_paras:
                            param.requires_grad = (adapterMode == 'train')
                        for param in trainable_a_paras:
                            param.requires_grad = (adapterMode == 'switch')
                        for param in trainable_v_paras:
                            param.requires_grad = (adapterMode == 'switch')

                    global_step = pbar.n * (1 + (self.stage == 1)) + (1 + mode_id) + delta_step * (self.stage == 2)

                    if adapterMode == 'train':
                        curr_bids, last_bid = \
                            self.train_weight(
                                data = all_data[adapterMode],
                                train_split = train_split,
                                batch_id = batch_id,
                                w_optim = w_optimizer,
                                scheduler = scheduler,
                                trainable_w_paras = trainable_w_paras,
                                global_step = global_step,
                                gradient_accumulate_steps = gradient_accumulate_steps,
                                tqdm_file = tqdm_file,
                                records = records,
                                specaug = specaug
                            )
                    else:
                        curr_bids, last_bid = \
                            self.train_arch(
                                input_data = all_data,
                                train_split = train_split,
                                batch_id = batch_id,
                                w_optim = w_optimizer,
                                a_optim = a_optimizer,
                                v_optim = v_optimizer,
                                trainable_w_paras = trainable_w_paras,
                                trainable_a_paras = trainable_a_paras,
                                trainable_v_paras = trainable_v_paras,
                                global_step = global_step,
                                gradient_accumulate_steps = gradient_accumulate_steps,
                                records = records,
                                specaug = specaug
                            )
                    
                    batch_ids += curr_bids
                    batch_id = last_bid

                if self.stage < 2 and self.adapter_config.adapter.switch.tau.type != "const":
                    self.upstream.model.reduce_tau()
                
                pbar.update(1)
                
                if not is_leader_process():
                    batch_ids = []
                    records = defaultdict(list)
                    continue

                # logging
                if global_step % self.config['runner']['log_step'] == 0 or global_step + 1 == self.config['runner']['total_steps']:
                    layers, norm_weights = self.upstream.model.get_layers, self.featurizer.model.get_norm_weights
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
                        to_wandb = self.args.online
                    )
                    batch_ids = []
                    records = defaultdict(list)

                # evaluation and save checkpoint
                save_names = []

                if global_step % self.config['runner']['eval_step'] == 0 or global_step + 1 == self.config['runner']['total_steps']:
                    for split in self.config['runner']['eval_dataloaders']:
                        save_names += self.evaluate(split, logger, global_step)

                if global_step % self.config['runner']['save_step'] == 0 or global_step + 1 == self.config['runner']['total_steps']:
                    def check_ckpt_num(directory):
                        max_keep = self.config['runner']['max_keep']
                        ckpt_pths = glob.glob(f'{directory}/states-*.ckpt')
                        if len(ckpt_pths) >= max_keep:
                            ckpt_pths = sorted(ckpt_pths, key=lambda pth: int(pth.split('-')[-1].split('.')[0]))
                            for ckpt_pth in ckpt_pths[:len(ckpt_pths) - max_keep + 1]:
                                os.remove(ckpt_pth)
                    check_ckpt_num(self.args.expdir)
                    save_names.append(f'states-{global_step}.ckpt')

                    # Dump current structure to folder 'exp_name'
                    with open(os.path.join(self.args.expdir, 'architecture.json'), 'w') as f:
                        architect = {}
                        for layer in self.upstream.model.get_layers:
                            architect.update({layer.adapterswitch.layer_idx: [layer.adapterswitch.used_adapter[idx] for idx in layer.adapterswitch.fixed_idx]})
                        architect = json.dumps(architect, indent=4)
                        print(architect, file=f)

                if len(save_names) > 0:
                    all_states = {
                        'Optimizer': {
                            "w_optimizer": w_optimizer.state_dict(), 
                            "a_optimizer": a_optimizer.state_dict(),
                            "v_optimizer": v_optimizer.state_dict() if v_optimizer else None
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
                            named_paras = entry.model.get_named_parameters
                            adapter_state = {}
                            for name, param in named_paras:
                                if getattr(param, '__is_delta__', False) or getattr(param, '__is_switch__', False):
                                    adapter_state[name] = param
                            all_states["adapter"] = adapter_state

                    if scheduler:
                        all_states['Scheduler'] = scheduler.state_dict()

                    if is_initialized():
                        all_states['WorldSize'] = get_world_size()

                    save_paths = [os.path.join(self.args.expdir, name) for name in save_names]
                    tqdm.write(f'[Runner] - Save the checkpoint to:')
                    for i, path in enumerate(save_paths):
                        tqdm.write(f'{i + 1}. {path}')
                        torch.save(all_states, path)
                    del all_states
      
        pbar.close()
        self.inner_pbar.close()

        if self.args.push_to_hf_hub:
            self.push_to_huggingface_hub()
        if is_leader_process():
            logger.close()
            wandb.finish()

    def model_forward(self, data: dict, train_split, records, specaug=None, use_last=True, return_predicted=False):
        # forward data to the whole pipeline
        if self.adapter_config.adapter.switch.algo.name == 's3delta' and not self.adapter_config.adapter.switch.baseline:
            self.upstream.model.compute_shifted_sigmoid(
                max_num_param=self.adapter_config.adapter.switch.algo.para_budget, 
                tau=self.adapter_config.adapter.switch.algo.sigmoid_tau,
                use_last=use_last
            )
        wavs = [torch.FloatTensor(wav).to(self.args.device) for wav in data['wavs']]
        if self.upstream.trainable:
            features = self.upstream.model(wavs)
        else:
            with torch.no_grad():
                features = self.upstream.model(wavs)

        features = self.featurizer.model(wavs, features)
        if specaug:
            features, _ = specaug(features)
        
        output = self.downstream.model(
            train_split,
            features, *data['others'],
            records = records,
            return_predicted = return_predicted # If set to True, then Downstream will return output_logits, rather than loss
        )
        return output

    def use_virtual(self):
        for entry in self.all_entries:
            entry.model.use_virtual()

    def use_default(self):
        for entry in self.all_entries:
            entry.model.use_default()

    def train_arch(
            self,
            input_data = {},
            train_split = None,
            batch_id = 0,
            w_optim = None,
            a_optim = None,
            v_optim = None,
            trainable_w_paras = [],
            trainable_a_paras = [],
            trainable_v_paras = [],
            global_step = 0,
            gradient_accumulate_steps = 1,
            records = defaultdict(list),
            specaug = None
        ):
        batch_ids = []
        if self.adapter_config.adapter.switch.algo.use_gumbel and self.stage < 2:
            self.upstream.model.sample_gumbel()
        if self.adapter_config.adapter.switch.algo.name == 's3delta':
            self.upstream.model.sample_uniform()
        try:
            if self.do_virtual:
                v_optim.load_state_dict(w_optim.state_dict())
                self.use_virtual()
                # calculate virtual loss on virtual model for one step
                v_records = defaultdict(list)
                
                for p in trainable_a_paras:
                    p.requires_grad = False
                
                for i in range(gradient_accumulate_steps):
                    v_loss = self.model_forward(input_data['train'][i], train_split, v_records, specaug, use_last=False)
                    (v_loss / gradient_accumulate_steps).backward()
                    del v_loss
                
                # update vitrual parameters
                v_optim.step()
                v_optim.zero_grad()

                # calculate architecture loss using virtual_model with current architecture weight
                for p in trainable_a_paras:
                    p.requires_grad = True
                
                all_dalpha = []
                all_dw = []
                curr_trainable_parameters = trainable_a_paras + trainable_v_paras if self.adapter_config.adapter.switch.algo.second_order else trainable_a_paras
                
                for i in range(gradient_accumulate_steps):
                    a_loss = self.model_forward(input_data['switch'][i], train_split, records, specaug)
                    grad = torch.autograd.grad(
                        (a_loss/ gradient_accumulate_steps), curr_trainable_parameters, allow_unused=True
                    )
                    del a_loss
                    
                    all_dalpha.append(grad[:len(trainable_a_paras)])
                    if self.adapter_config.adapter.switch.algo.second_order:
                        all_dw.append(grad[len(trainable_a_paras):])

                    batch_id += 1
                    batch_ids.append(batch_id)

                self.use_default()

                summed_dalpha = []
                for dalpha in zip(*all_dalpha):
                    valid_da = [da for da in dalpha if da is not None]
                    summed_da = sum(valid_da) if valid_da else None
                    summed_dalpha.append(summed_da)
                
                if self.adapter_config.adapter.switch.algo.second_order:
                    summed_dw = []
                    for dw in zip(*all_dw):
                        valid_dw = [w for w in dw if w is not None]
                        summed_w = sum(valid_dw) if valid_dw else None
                        summed_dw.append(summed_w)
                    # compute the higher-order derivatives
                    all_hessian = self.compute_hessian(trainable_w_paras, trainable_a_paras, summed_dw, input_data['train'], train_split, gradient_accumulate_steps, specaug)
                    with torch.no_grad():
                        for alpha, dalpha, hessian in zip(trainable_a_paras, summed_dalpha, all_hessian):
                            alpha.grad = dalpha - w_optim.param_groups[0]['lr'] * hessian
                else:
                    # First order approximation
                    with torch.no_grad():
                        for alpha, dalpha in zip(trainable_a_paras, summed_dalpha):
                            alpha.grad = dalpha
            else:
                # normal forward
                for i in range(gradient_accumulate_steps):
                    loss = self.model_forward(input_data['switch'][i], train_split, records, specaug, use_last=False)
                    (loss / gradient_accumulate_steps).backward()
                    del loss

                    batch_id += 1
                    batch_ids.append(batch_id)
                
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                linelogger.info(f'[Runner] - CUDA out of memory at step {global_step}')
            raise
        
        if self.adapter_config.adapter.switch.algo.name == 'fair_darts':
            aux_loss = self.upstream.model.aux_loss() * self.adapter_config.adapter.switch.algo.aux_loss_ratio
            aux_loss.backward()
            del aux_loss

        # gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable_a_paras, self.config['runner']['gradient_clipping'])

        # optimize
        if math.isnan(grad_norm):
            linelogger.info(f'[Runner] - grad norm is NaN at step {global_step}')
        else:
            # update architecture weight
            a_optim.step()
            a_optim.zero_grad()

        return batch_ids, batch_id

    def compute_hessian(
            self, 
            trainable_w_paras, 
            trainable_a_paras, 
            dw,
            train_data,
            train_split,
            gradient_accumulate_steps,
            specaug
        ):
        """
        dw = dw` { L_val(w`, alpha) }
        w+ = w + eps * dw
        w- = w - eps * dw
        hessian = (dalpha { L_trn(w+, alpha) } - dalpha { L_trn(w-, alpha) }) / (2*eps)
        eps = 0.01 / ||dw||
        """
        original_w_paras = deepcopy(trainable_w_paras)
        norm = torch.cat([w.view(-1) for w in dw if w is not None]).norm()
        if norm == 0:
            print('norm is zero')
        eps = 0.01 / (norm + 1e-6)

        # w+ = w + eps*dw`
        with torch.no_grad():
            for p, d in zip(trainable_w_paras, dw):
                if d is not None:
                    p += eps * d

        all_dalpha_pos = []
        records = defaultdict(list)
        for i in range(gradient_accumulate_steps):
            loss1 = self.model_forward(train_data[i], train_split, records, specaug)
            all_dalpha_pos.append(
                torch.autograd.grad(loss1 / gradient_accumulate_steps, trainable_a_paras)
            )  # dalpha { L_trn(w+) }
            del loss1

        # w- = w - eps*dw`
        with torch.no_grad():
            for p, op, d in zip(trainable_w_paras, original_w_paras, dw):
                p.data.copy_(op.data)
                if d is not None:
                    p -= eps * d
    
        all_dalpha_neg = []
        for i in range(gradient_accumulate_steps):
            loss2 = self.model_forward(train_data[i], train_split, records, specaug)
            all_dalpha_neg.append(
                torch.autograd.grad(loss2 / gradient_accumulate_steps, trainable_a_paras)
            )  # dalpha { L_trn(w-) }
            del loss2

        # recover w
        with torch.no_grad():
            for p, op in zip(trainable_w_paras, original_w_paras):
                p.data.copy_(op.data)

        del original_w_paras

        hessian = [
            [(p-n) / (2*eps) for p, n in zip(dalpha_pos, dalpha_neg)]
                for dalpha_pos, dalpha_neg in zip(all_dalpha_pos, all_dalpha_neg)
        ]

        result = []
        for h in zip(*hessian):
            summed_h = torch.zeros_like(h[0])
            for h_ in h:
                summed_h += h_
            result.append(summed_h)
        
        return result

    def train_weight(
            self,
            data = None,
            train_split = None,
            batch_id = 0,
            w_optim = None,
            scheduler = None,
            trainable_w_paras = [],
            global_step = 0,
            gradient_accumulate_steps = 1,
            tqdm_file = None,
            records = defaultdict(list),
            specaug = None
        ):
        try:
            batch_ids = []
            if self.adapter_config.adapter.switch.algo.use_gumbel and self.stage < 2:
                    self.upstream.model.sample_gumbel()
            for i in range(gradient_accumulate_steps):
                loss = self.model_forward(data[i], train_split, records, specaug)

                batch_id += 1
                batch_ids.append(batch_id)

                (loss / gradient_accumulate_steps).backward()
                del loss

                # Update progress bar
                self.inner_pbar.update(1)
                if self.inner_pbar.n == self.inner_pbar.total:
                    total = self.inner_pbar.total
                    self.inner_pbar.close()
                    self.inner_pbar = tqdm(total=total, dynamic_ncols=True, desc=f'train_stage{self.stage}', file=tqdm_file)
                    batch_id = -1

        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                linelogger.info(f'[Runner] - CUDA out of memory at step {global_step}')
            raise

        # gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable_w_paras, self.config['runner']['gradient_clipping'])

        # optimize
        if math.isnan(grad_norm):
            linelogger.info(f'[Runner] - grad norm is NaN at step {global_step}')
        else:
            w_optim.step()    
            w_optim.zero_grad()
        
        # adjust learning rate
        if scheduler:
            scheduler.step()

        return batch_ids, batch_id
    
    def synflow(self, split=None, logger=None):
        # fix seed to guarantee the same evaluation protocol across steps 
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        
        if self.adapter_config.adapter.switch.algo.name == 's3delta':
            self.upstream.model.set_hard_forward_structure(max_num_param=self.adapter_config.adapter.switch.algo.para_budget, baseline=self.adapter_config.adapter.switch.baseline)

        for entry in self.all_entries:
            # set the sign of parameters to positive
            for name, param in entry.model.named_parameters():
                setattr(param, 'cal_synflow', True)
                print(f'cal synflow on {name}')
                param.requires_grad = False
                param.abs_()
                param.requires_grad = True

        train_split = self.config['runner'].get("train_dataloader", "train")
        records = defaultdict()
        
        data = {'wavs': [[1 for _ in range(400)]], 'others': [np.array([[0]]), [400], 0]}
        out = self.model_forward(data, train_split, records, return_predicted=True)
        torch.sum(out).backward()

        sum_grads = 0
        for entry in self.all_entries:
            for name, param in entry.model.named_parameters():
                if getattr(param, 'cal_synflow', False) and param.grad is not None:
                    sum_grads += torch.sum(torch.abs(param * param.grad))
        
        linelogger.info(f'synflow: {sum_grads}')
        with open('synflow.csv', 'a') as syn_f:
            print(f'{self.args.rand_seq},{sum_grads}', file=syn_f)
        if self.args.online:
            wandb.log({'synflow': sum_grads})

        return

    def evaluate(self, split=None, logger=None, global_step=0):
        """evaluate function will always be called on a single process even during distributed training"""

        # When this member function is called directly by command line
        not_during_training = split is None and logger is None and global_step == 0
        if not_during_training:
            split = self.args.evaluate_split
            tempdir = tempfile.mkdtemp()
            logger = SummaryWriter(tempdir)

        if self.adapter_config.adapter.switch.algo.name == 's3delta':
            self.upstream.model.set_hard_forward_structure(max_num_param=self.adapter_config.adapter.switch.algo.para_budget, baseline=self.adapter_config.adapter.switch.baseline)

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
        dataloader = self.downstream.model.get_dataloader(split)
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
