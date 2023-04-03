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

import torch
import torchaudio
import numpy as np
import wandb
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import is_initialized, get_rank, get_world_size

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
linelogger = logging.getLogger(__name__)
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT)
linelogger.setLevel(logging.INFO)
from ..upstream.adapterModels import dict2obj, MyLogger
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

        if isinstance(args.upstream_adapter_config, str):   # In evaluate mode (i.e. run with -e evaluate), this parameter will be dict.
            with open(args.upstream_adapter_config, 'r') as file:
                self.adapterDict = yaml.load(file, Loader=yaml.FullLoader)
        else:
            self.adapterDict = args.upstream_adapter_config
        
        if self.args.mode == 'train':
            # train both stage1 and stage2
            self.stage1_steps = self.args.stage1_ratio * self.config['runner']['total_steps'] // 2
            self.stage2_steps = self.config['runner']['total_steps'] - self.stage1_steps
        elif self.args.mode == 'train_stage1':
            self.stage1_steps = self.config['runner']['total_steps']
            self.stage2_steps = 0
        elif self.args.mode == 'train_stage2':
            self.stage1_steps = 0
            self.stage2_steps = self.config['runner']['total_steps']
        else:
            self.stage1_steps = self.stage2_steps = 0

        self.adapter_config = dict2obj(self.adapterDict)
        self.adapter_config.adapter.switch.tau.steps = self.config['runner']['total_steps']
        self.adapter_config.adapter.switch.tau.init_steps = self.init_ckpt.get('Step', 0)
        self.adapter_config.adapter.switch.stage = 2 if self.args.mode == 'train_stage2' else 1
        
        print(f'stage: {self.adapter_config.adapter.switch.stage}')
        linelogger.info(f"{self.adapter_config.adapter.switch.tau.init_steps}")
        linelogger.info(f"{self.adapter_config.adapter.switch.tau.steps}, {self.adapter_config.adapter.switch.tau.stop_value}")
        
        self.upstream = self._get_upstream()
        self.featurizer = self._get_featurizer()
        self.downstream = self._get_downstream()
        self.all_entries = [self.upstream, self.featurizer, self.downstream]


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
                upstream_weight = self.stage2_ckpt.get("Upstream")
                if upstream_weight:
                    show(f'[Runner] - Loading {"Adapter"} weights & switch logits from the stage1 experiment')
                    model_dict = model.state_dict()
                    for para, value in upstream_weight.items():
                        if 'adapter' in para:
                            model_dict.update({para: value})
                            show(f'{para}: {value}')
                    model.load_state_dict(model_dict)

                upstream_weight = self.init_ckpt.get('Upstream')
                if upstream_weight:
                    # init ckpt is given
                    assert self.stage2_ckpt == {}, \
                        "init_ckpt and stage2_ckpt shouldn't be provided simultaneously."
                    
                    model_dict = model.state_dict()
                    for para, value in upstream_weight.items():
                        if 'adapter' in para:
                            model_dict.update({para: value})
                    
                    show(f'[Runner] - Loading {"Adapter"} weights from the previous experiment')
                    model.load_state_dict(model_dict)
        elif name == 'Downstream':
            downstream_weight = self.stage2_ckpt.get('Downstream')
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


    def _get_scheduler(self, optimizer):
        scheduler = get_scheduler(
            optimizer,
            self.config['runner']['total_steps'],
            self.config['scheduler']
        )
        self._load_weight(scheduler, 'Scheduler')
        return scheduler

    def _create_model_card(self, path):
        model_card = MODEL_CARD_MARKDOWN.format(upstream_model=self.args.upstream)
        with open(os.path.join(path, "README.md"), "w") as f:
            f.write(model_card)

    def train_stage1(self):
        if is_leader_process():
            wandb.init(project=f'{self.args.upstream}-{self.args.downstream}')
            newArg = self.args
            newArg.config = self.config
            newArg.upstream_adapter_config = self.adapterDict
            wandb.config.update(newArg)
            wandb.define_metric("dev-per", summary="min")
            wandb.define_metric("dev-loss", summary="min")
            wandb.define_metric("train-per", summary="min")
            wandb.define_metric("train-loss", summary="min")
        # trainable parameters and train/eval mode
        trainable_paras = []
        # Network weights
        trainable_w_paras = []
        # Architecture weights (switch)
        trainable_a_paras = []
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
                for name, param in entry.model.named_parameters():
                    if "adapter" in name or 'lora' in name:
                        param.requires_grad = True
                        if 'switch' in name:
                            trainable_a_paras.append(param)
                        else:
                            trainable_w_paras.append(param)
                    else:
                        param.requires_grad = False
                    
                trainable_paras += list(additional_weight)       
            elif entry.trainable:
                if entry.name == 'Featurizer' and (self.stage1_steps <= 0 or not self.args.stage1_weighted_sum):
                    # Not to train weighted sum in stage1
                    entry.model.eval()
                linelogger.info(f"append weights: {entry.name}, {len(list(entry.model.parameters()))}")
                trainable_w_paras += list(entry.model.parameters())
            else:
                print(f'in eval: {entry.name}')
                entry.model.eval()

        # optimizer
        w_optimizer = self._get_optimizer(trainable_w_paras, 'w_optimizer', [])
        a_optimizer = self._get_optimizer(trainable_a_paras, 'a_optimizer', [])

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

        backward_steps = 0
        batch_ids = []
        records = defaultdict(list)
        epoch = self.init_ckpt.get('Epoch', 0)        
        train_split = self.config['runner'].get("train_dataloader", "train")

        if self.stage1_steps > 0:
            linelogger(f'train stage1 for {self.stage1_steps} steps')
            adapterModes = ['train', 'switch'] if len(self.adapter_config.adapter.switch.path) > 1 and \
                                                    not self.adapter_config.adapter.switch.first else ['switch', 'train']            
            try:
                dataloaders = self.downstream.model.get_dataloader(train_split, 'train_stage1', epoch=epoch)
            except TypeError as e:
                if "unexpected keyword argument 'epoch'" in str(e):
                    try:
                        dataloaders = self.downstream.model.get_dataloader(train_split, 'train_stage1')
                        for adapterMode in adapterModes:
                            if hasattr(dataloaders[adapterMode], "sampler") and isinstance(dataloaders[adapterMode].sampler, DistributedSampler):
                                dataloaders[adapterMode].sampler.set_epoch(epoch)
                    except:
                        raise
                else:
                    raise
            
            for adapterMode in adapterModes:
                linelogger.info(f'dataset size of {adapterMode}: {len(dataloaders[adapterMode].dataset)}')
            for adapterMode in adapterModes:
                linelogger.info(f'data loader size of {adapterMode}: {len(dataloaders[adapterMode])}')
                linelogger.info(f'dataset # indice of {adapterMode}: {len(dataloaders[adapterMode].dataset.indices)}')
            linelogger.info(f'dataset overlap: {len(set(dataloaders["train"].dataset.indices) & set(dataloaders["switch"].dataset.indices))}')

            input_modes, cur_step, iters = {}, {}, {}
            for adapterMode in adapterModes:
                input_modes[adapterMode] = None
                cur_step[adapterMode] = 0
                iters[adapterMode] = iter(dataloaders[adapterMode])

            # Log initial tau, switch logits & norm_weight to wandb
            if is_leader_process():
                results = {}
                for i, layer in enumerate(self.upstream.model.module.model.encoder.layers):
                    for j, logit in enumerate(list(layer.adapterswitch.probs.cpu())):
                        results.update({f"layer_{i}/{train_split}_{j}": logit.item()})
                    results.update({f"tau": layer.adapterswitch.switch_temperature[0]})
                
                for i, weight in enumerate(self.featurizer.model.norm_weights):
                    results.update({f"{train_split}_norm_weights_{i}": weight})

                results.update({"lr": scheduler.get_last_lr()[0]})
                wandb.log(results, step=pbar.n)

                del results
            linelogger.info(f"gradient accumulate steps: {self.config['runner'].get('gradient_accumulate_steps')}")
            while pbar.n < self.stage1_ckpt:
                for batch_id, (wavs, *others) in enumerate(tqdm(dataloaders['train'], dynamic_ncols=True, desc='train_stage1', file=tqdm_file)):
                    if pbar.n >= pbar.total:
                        break
                    try:
                        (valid_wavs, *valid_others) = next(iters['switch'])
                    except StopIteration:
                        iters['switch'] = iter(dataloaders['switch'])
                        (valid_wavs, *valid_others) = next(iters['switch'])
                    
                    input_modes['train'] = {'wavs': wavs, 'others': others, 'add_weight': []}
                    input_modes['switch'] = {'wavs': valid_wavs, 'others': valid_others, 'add_weight': []}

                    for adapterMode in adapterModes:
                        optimizer, lr_scheduler, trainable_paras = \
                            (w_optimizer, scheduler, trainable_w_paras) if adapterMode == 'train' else (a_optimizer, None, trainable_a_paras)
                        
                        for entry in self.all_entries:
                            if self.args.adapter != False and entry.name == "Upstream":
                                for name, param in entry.model.named_parameters():
                                    if "adapter" in name or 'lora' in name:
                                        param.requires_grad = ("switch" in name) ^ (adapterMode == "train")
                                        if param.requires_grad:
                                            # linelogger.info(name)
                                            input_modes[adapterMode]['add_weight'].append(param)
                                        # print("Adapter!!", name, param.requires_grad)
                                    else:
                                        param.requires_grad = False
                            # if entry.name == "Featurizer":
                            #     for name, param in entry.model.named_parameters():
                            #         param.requires_grad = (adapterMode == "train")
                        try:
                            global_step = pbar.n + 1
                            wavs = [torch.FloatTensor(wav).to(self.args.device) for wav in input_modes[adapterMode]['wavs']]
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
                            batch_ids.append(batch_id * 2 + (adapterMode == 'switch'))

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
                        optimizer.zero_grad()

                        # adjust learning rate
                        if lr_scheduler:
                            lr_scheduler.step()
                    
                    if backward_steps % gradient_accumulate_steps > 0:
                        continue
                    
                    self.upstream.model.module.model.reduce_tau()
                    if not is_leader_process():
                        batch_ids = []
                        records = defaultdict(list)
                        continue

                    # logging
                    if global_step % self.config['runner']['log_step'] == 0:
                        self.downstream.model.log_records(
                            train_split,
                            records = records,
                            logger = logger,
                            global_step = global_step,
                            batch_ids = batch_ids,
                            total_batch_num = sum([len(dataloaders[m]) for m in adapterModes]),
                            adapter_mode = adapterMode,
                            layers = self.upstream.model.module.model.encoder.layers,  # add module after first model
                            norm_weights = self.featurizer.model.norm_weights.detach(),
                            lr = scheduler.get_last_lr()[0],
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
                            'Optimizer': {"w_optimizer": w_optimizer.state_dict(), "a_optimizer": a_optimizer.state_dict()},
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
                                    if "adapter" in name:
                                        adapter_state[name] = param
                                    if self.args.adapter == "bitfit":
                                        if "bias" in name:
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

                    pbar.update(1)
                    
                epoch += 1
                for adapterMode in adapterModes:
                    if hasattr(dataloaders[adapterMode], "sampler") and isinstance(dataloaders[adapterMode].sampler, DistributedSampler):
                                dataloaders[adapterMode].sampler.set_epoch(epoch)
        
        if self.stage2_steps > 0:
            linelogger(f'train stage2 for {self.stage2_steps} steps')
            if self.args.stage2_weighted_sum:
                # enable weighted sum in stage2
                for entry in self.all_entries:
                    if entry.name == 'Featurizer':
                        entry.model.train()
            # change switch stage to 2 to perform one-hot forwarding
            self.upstream.model.module.model.set_stage(2)

            if is_leader_process():
                results = {}
                for i, layer in enumerate(self.upstream.model.module.model.encoder.layers):
                    for j, logit in enumerate(list(layer.adapterswitch.probs.cpu())):
                        results.update({f"layer_{i}/{train_split}_{j}": logit.item()})
                    results.update({f"tau": layer.adapterswitch.switch_temperature[0]})
                
                for i, weight in enumerate(self.featurizer.model.norm_weights):
                    results.update({f"{train_split}_norm_weights_{i}": weight})

                results.update({"lr": scheduler.get_last_lr()[0]})
                wandb.log(results, step=pbar.n)

                del results

            while pbar.n < pbar.total:
                try:
                    dataloaders = self.downstream.model.get_dataloader(train_split, 'train_stage2', epoch=epoch)
                except TypeError as e:
                    if "unexpected keyword argument 'epoch'" in str(e):
                        dataloaders = self.downstream.model.get_dataloader(train_split)
                        if hasattr(dataloaders, "sampler") and isinstance(dataloaders.sampler, DistributedSampler):
                            dataloaders.sampler.set_epoch(epoch)
                    else:
                        raise
                for batch_id, (wavs, *others) in enumerate(tqdm(dataloaders['train'], dynamic_ncols=True, desc='train_stage2', file=tqdm_file)):
                    # try/except block for forward/backward
                    try:
                        if pbar.n >= pbar.total:
                            break
                        global_step = pbar.n + 1
                        wavs = [torch.FloatTensor(wav).to(self.args.device) for wav in wavs]
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
                            features, *others,
                            records = records,
                        )
                        batch_ids.append(batch_id)

                        gradient_accumulate_steps = self.config['runner'].get('gradient_accumulate_steps')
                        (loss / gradient_accumulate_steps).backward()
                        del loss, wavs, features

                    except RuntimeError as e:
                        if 'CUDA out of memory' in str(e):
                            print(f'[Runner] - CUDA out of memory at step {global_step}')
                            if is_initialized():
                                raise
                            with torch.cuda.device(self.args.device):
                                torch.cuda.empty_cache()
                            optimizer.zero_grad()
                            continue
                        else:
                            raise

                    # whether to accumulate gradient
                    backward_steps += 1
                    if backward_steps % gradient_accumulate_steps > 0:
                        continue

                    # gradient clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        trainable_paras, self.config['runner']['gradient_clipping'])

                    # optimize
                    if math.isnan(grad_norm):
                        print(f'[Runner] - grad norm is NaN at step {global_step}')
                    else:
                        optimizer.step()
                    optimizer.zero_grad()

                    # adjust learning rate
                    if scheduler:
                        scheduler.step()

                    if not is_leader_process():
                        batch_ids = []
                        records = defaultdict(list)
                        continue

                    # logging
                    if global_step % self.config['runner']['log_step'] == 0:
                        self.downstream.model.log_records(
                            train_split,
                            records = records,
                            logger = logger,
                            global_step = global_step,
                            batch_ids = batch_ids,
                            total_batch_num = len(dataloaders['train']),
                            layers = self.upstream.model.module.model.encoder.layers,  # add module after first model
                            norm_weights = self.featurizer.model.module.norm_weights,
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
                            'Optimizer': optimizer.state_dict(),
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
                                        if "adapter" in name:
                                            adapter_state[name] = param
                                        if self.args.adapter == "bitfit":
                                            if "bias" in name:
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

                    pbar.update(1)
                epoch += 1
                    
        pbar.close()

        if self.args.push_to_hf_hub:
            self.push_to_huggingface_hub()
        if is_leader_process():
            logger.close()
            wandb.finish()

    def train_stage2(self):
        if is_leader_process():
            wandb.init(project=f'{self.args.upstream}-{self.args.downstream}')
            newArg = self.args
            newArg.config = self.config
            newArg.upstream_adapter_config = self.adapterDict
            wandb.config.update(newArg)
            wandb.define_metric("dev-per", summary="min")
            wandb.define_metric("dev-loss", summary="min")
            wandb.define_metric("train-per", summary="min")
            wandb.define_metric("train-loss", summary="min")
        
        # trainable parameters and train/eval mode
        trainable_models = []
        trainable_paras = []
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

            #### add adapters ##################
            if self.args.adapter != None and entry.name == "Upstream":
                adapter_param = 0
                for  name, param in entry.model.named_parameters():
                        if "adapter" in name or 'lora' in name:
                            additional_weight.append(param)
                            param.requires_grad = True
                            print("Adapter!!",name)
                            adapter_param += param.nelement() 
                            #print("Numbers of PARAMETER: %.2fM" % (total/1e6))
                        else:
                            param.requires_grad = False
                    
                trainable_paras += list(additional_weight)
                print("total_adapter param")
                print("Numbers of adapter PARAMETER: %.2fM" % (adapter_param/1e6))

            if entry.trainable:
                entry.model.train()
                trainable_models.append(entry.model)
                trainable_paras += list(entry.model.parameters())
            else:
                entry.model.eval()

        # optimizer
        optimizer = self._get_optimizer(trainable_models, "Optimizer", [])

        # scheduler
        scheduler = None
        if self.config.get('scheduler'):
            scheduler = self._get_scheduler(optimizer)

        # specaug
        specaug = None
        if self.config.get('specaug'):
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

        batch_ids = []
        backward_steps = 0
        records = defaultdict(list)
        epoch = self.init_ckpt.get('Epoch', 0)
        train_split = self.config['runner'].get("train_dataloader", "train")

        # Log switch logits & norm_weight from stage1_ckpt to wandb
        if is_leader_process():
            results = {}
            for i, layer in enumerate(self.upstream.model.module.model.encoder.layers):
                for j, logit in enumerate(list(layer.adapterswitch.probs.cpu())):
                    results.update({f"layer_{i}/{train_split}_{j}": logit.item()})
                results.update({f"tau": layer.adapterswitch.switch_temperature[0]})
            print(self.featurizer.model.module.norm_weights)
            for i, weight in enumerate(self.featurizer.model.module.norm_weights):
                results.update({f"{train_split}_norm_weights_{i}": weight})
            wandb.log(results, step=pbar.n)
            del results

        while pbar.n < pbar.total:
            try:
                dataloaders = self.downstream.model.get_dataloader(train_split, self.args.mode, epoch=epoch)
                # print(f'mode: {self.args.mode}')
            except TypeError as e:
                if "unexpected keyword argument 'epoch'" in str(e):
                    dataloaders = self.downstream.model.get_dataloader(train_split)
                    if hasattr(dataloaders, "sampler") and isinstance(dataloaders.sampler, DistributedSampler):
                        dataloaders.sampler.set_epoch(epoch)
                else:
                    raise
            for batch_id, (wavs, *others) in enumerate(tqdm(dataloaders['train'], dynamic_ncols=True, desc='train', file=tqdm_file)):
                # try/except block for forward/backward
                try:
                    if pbar.n >= pbar.total:
                        break
                    global_step = pbar.n + 1
                    wavs = [torch.FloatTensor(wav).to(self.args.device) for wav in wavs]
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
                        features, *others,
                        records = records,
                    )
                    batch_ids.append(batch_id)

                    gradient_accumulate_steps = self.config['runner'].get('gradient_accumulate_steps')
                    (loss / gradient_accumulate_steps).backward()
                    del loss, wavs, features

                except RuntimeError as e:
                    if 'CUDA out of memory' in str(e):
                        print(f'[Runner] - CUDA out of memory at step {global_step}')
                        if is_initialized():
                            raise
                        with torch.cuda.device(self.args.device):
                            torch.cuda.empty_cache()
                        optimizer.zero_grad()
                        continue
                    else:
                        raise

                # whether to accumulate gradient
                backward_steps += 1
                if backward_steps % gradient_accumulate_steps > 0:
                    continue

                # gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_paras, self.config['runner']['gradient_clipping'])

                # optimize
                if math.isnan(grad_norm):
                    print(f'[Runner] - grad norm is NaN at step {global_step}')
                else:
                    optimizer.step()
                optimizer.zero_grad()

                # adjust learning rate
                if scheduler:
                    scheduler.step()

                if not is_leader_process():
                    batch_ids = []
                    records = defaultdict(list)
                    continue

                # logging
                if global_step % self.config['runner']['log_step'] == 0:
                    self.downstream.model.log_records(
                        train_split,
                        records = records,
                        logger = logger,
                        global_step = global_step,
                        batch_ids = batch_ids,
                        total_batch_num = len(dataloaders['train']),
                        layers = self.upstream.model.module.model.encoder.layers,  # add module after first model
                        norm_weights = self.featurizer.model.module.norm_weights,
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
                        'Optimizer': optimizer.state_dict(),
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
                                    if "adapter" in name:
                                        adapter_state[name] = param
                                    if self.args.adapter == "bitfit":
                                        if "bias" in name:
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

                pbar.update(1)
            epoch += 1

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
