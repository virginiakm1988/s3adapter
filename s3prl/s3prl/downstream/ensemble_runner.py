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
from s3prl.utility.helper import is_leader_process, get_model_state, show, defaultdict, override

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
from s3prl.upstream.adapterModels import dict2obj, MyLogger, is_baseline, find_module
from typing import List, Dict
import functools
import ray
import itertools
# from darts
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

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)




class ModelEntry:
    def __init__(self, model, name, trainable, interfaces):
        self.model = model
        self.name = name
        self.trainable = trainable
        self.interfaces = interfaces


class EnsembleRunner():
    """
    Used to handle high-level concepts of a ML experiment
    eg. training loop, evaluation loop, upstream propagation, optimization, logging, checkpoint saving
    """
    def __init__(self, args, config):
        self.upstreams = []
        self.featurizers = []
        self.downstreams = []
        for adapter_ckpt in args.ensemble:
            args.init_ckpt = adapter_ckpt
            ckpt_dir = os.path.dirname(adapter_ckpt)
            args_cfg_files = glob.glob(f'{ckpt_dir}/args_*.yaml')
            args_cfg_files.sort(key=lambda x: os.path.getmtime(x))
            with open(args_cfg_files[-1], 'r') as file:
                args_cfg = yaml.load(file, Loader=yaml.FullLoader)
                override(args_cfg['override'], args, config)
            self.args = args
            show(self.args)
            self.config = config

            self.init_ckpt = torch.load(self.args.init_ckpt, map_location='cpu') if self.args.init_ckpt else {}
            self.exchange_adapter = torch.load(self.args.exchange_adapter, map_location='cpu') if self.args.exchange_adapter else {}

            '''
            if isinstance(args.upstream_adapter_config, str):   # In evaluate mode (i.e. run with -e evaluate), this parameter will be dict.
                with open(args.upstream_adapter_config, 'r') as file:
                    self.adapterDict = yaml.load(file, Loader=yaml.FullLoader)
            else:
                self.adapterDict = args.upstream_adapter_config
            '''
            
            self.adapterDict = {'adapter': config['adapter_config']}
            self.adapter_config = dict2obj(self.adapterDict)
            
            self.prepare_baseline(num_layers=(12 if not self.args.upstream == 'hubert_large_ll60k' else 24))

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
            
            self.adapter_config.adapter.switch.algo.re_init = self.args.random_exp

            linelogger.info(f'do_virtual: {self.do_virtual}, stage = {self.stage}, switch.stage = {self.adapter_config.adapter.switch.stage}')
            if self.do_virtual:
                assert not (self.adapter_config.adapter.switch.algo.second_order and not self.adapter_config.adapter.switch.algo.first_order),\
                "Second order should be calculated when first order is enable."

            self.adapter_config.adapter.switch.algo.re_init = self.args.random_exp

            self.upstreams.append(self._get_upstream())
            self.featurizers.append(self._get_featurizer(self.upstreams[-1]))
            self.downstreams.append(self._get_downstream(self.featurizers[-1]))
        # self.upstreams = []
        # self.featurizers = []
        # self.downstreams = []
        # for adapter_ckpt in args.ensemble:
        #     args.init_ckpt = adapter_ckpt
        #     ckpt_dir = os.path.dirname(adapter_ckpt)
        #     args_cfg_files = glob.glob(f'{ckpt_dir}/args_*.yaml')
        #     args_cfg_files.sort(key=lambda x: os.path.getmtime(x))
        #     with open(args_cfg_files[-1], 'r') as file:
        #         args_cfg = yaml.load(file, Loader=yaml.FullLoader)
        #         override(args_cfg['override'], args, config)
        #     show(args)
        #     show(config)
        #     self.upstreams.append(self._get_upstream())
        #     self.featurizers.append(self._get_featurizer(self.upstreams[-1]))
        #     self.downstreams.append(self._get_downstream(self.featurizers[-1]))

        # Used for averaging the log_prob
        self.real_downstream = self._get_downstream(self.featurizers[-1])

        # init wandb
        if is_leader_process():
            wandb_name = f'{self.args.search_algo}, {int(self.args.stage1_ratio * 100)}% search, lr {self.config["optimizer"]["lr"]}' if not self.adapter_config.adapter.switch.baseline else f'{self.args.search_algo} retrain'
            if self.args.random_exp:
                wandb_name = f'random exp {self.args.rand_seq}, budget={self.adapter_config.adapter.switch.algo.para_budget}, lr_rescheduled, init by distribution 3'
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
            
            if self.args.exchange_adapter:
                # override adapter weight
                exchange_adapter = self.exchange_adapter.get('adapter')
                if exchange_adapter:
                    show(f'[Runner] - Loading {"Adapter"} weights & switch logits from the previous experiment')
                    model_dict = model.state_dict()
                    model_dict.update(exchange_adapter)
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


    def _get_featurizer(self, upstream):
        model = Featurizer(
            upstream = upstream.model,
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


    def _get_downstream(self, featurizer):
        expert = importlib.import_module(f"s3prl.downstream.{self.args.downstream}.expert")
        Downstream = getattr(expert, "DownstreamExpert")

        model = Downstream(
            upstream_dim = featurizer.model.output_dim,
            upstream_rate = featurizer.model.downsample_rate,
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
    
    def prepare_baseline(self, num_layers=12):
        if self.args.random_exp:
            with open(self.args.rand_arch) as arch_f:
                all_arch = json.load(arch_f)
                arch = all_arch[str(self.args.rand_seq)]
                baseline = [[] for _ in range(num_layers)]
                for layer_id, used_adapters in enumerate(arch):
                    baseline[int(layer_id)] = [
                        idx for idx, adapter_name in enumerate(self.adapter_config.adapter.type) 
                        if adapter_name in used_adapters
                    ]
        elif self.args.mode == 'evaluate' and self.adapter_config.adapter.switch.algo.name == 's3delta':
            assert self.init_ckpt, 'Should provide a checkpoint for evaluation'
            arch_path = os.path.join(os.path.dirname(self.args.init_ckpt), 'architecture.json')
            if os.path.exists(arch_path):
                baseline = [[] for _ in range(24)]
                with open(arch_path, 'r') as arch_f:
                    arch = json.load(arch_f)
                    for layer_id, used_adapters in arch.items():
                        baseline[int(layer_id)] = [
                            idx for idx, adapter_name in enumerate(self.adapter_config.adapter.type) 
                            if adapter_name in used_adapters
                        ]
        else:
            baseline = self.adapter_config.adapter.switch.baseline
        
        self.adapter_config.adapter.switch.baseline = is_baseline(baseline, num_layers)
    
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
        for upstream in self.upstreams:
            upstream.model.eval()
        for featurizer in self.featurizers:
            featurizer.model.eval()
        for downstream in self.downstreams:
            downstream.model.eval()
        self.real_downstream.model.eval()
        # trainings = []
        # for entry in self.all_entries:
        #     trainings.append(entry.model.training)
        #     entry.model.eval()

        # prepare data
        dataloader = self.real_downstream.model.get_dataloader(split)
        evaluate_ratio = float(self.config["runner"].get("evaluate_ratio", 1))
        # HACK
        evaluate_ratio = 0.01
        evaluate_steps = round(len(dataloader) * evaluate_ratio)

        inputs, outputs, labels = [], [], []
        all_hiddens = defaultdict(list)
        batch_ids = []
        # records = defaultdict(list)
        records = defaultdict(lambda: defaultdict(list))
        torch.set_num_threads(32)
        # results = self.parallel_monkey(dataloader, split, evaluate_steps, records, batch_ids)
        dtwDs = DtwDataset(dataloader, self.upstreams, self.downstreams, self.featurizers, self.real_downstream, split, records, self.args)
        dtwLoader = torch.utils.data.DataLoader(dtwDs, batch_size=1, shuffle=False, num_workers=16, collate_fn=lambda x: x, pin_memory=True)
        logging.warning("Monkey...")
        for batch_id, _batch_object in enumerate(dtwLoader):
            logging.warning(f"batch_obj: {_batch_object}")
            seqs_ = torch.stack([_b["seq"] for _b in _batch_object], dim=0)
            wavs_ = [_b["wav"] for _b in _batch_object]
            others_ = [_b["other"] for _b in _batch_object]  # (batch, 2)
        
            wavs = [torch.FloatTensor(wav).to(self.args.device) for wav in wavs_]
            with torch.no_grad():
                features = self.upstreams[-1].model(wavs)
                features = self.featurizers[-1].model(wavs, features)
                self.real_downstream.model(
                    split,
                    features, *zip(*others_),
                    records = records["real"],
                    batch_id = batch_id,
                    log_probs = seqs_
            )   
        
        '''
        for batch_id, (wavs, *others) in enumerate(tqdm(dataloader, dynamic_ncols=True, desc=split, total=evaluate_steps)):
            if batch_id > evaluate_steps:
                break

            wavs = [torch.FloatTensor(wav).to(self.args.device) for wav in wavs]
            all_log_probs = []
            with torch.no_grad():
                for upstream, featurizer, downstream, ens_ckpt in zip(self.upstreams, self.featurizers, self.downstreams, self.args.ensemble):
                    features = upstream.model(wavs)
                    features = featurizer.model(wavs, features)
                    exp_name = os.path.dirname(ens_ckpt)
                    log_probs = downstream.model(
                        split,
                        features, *others,
                        records = records[exp_name],
                        batch_id = batch_id,
                        return_log_probs = True
                    )
                    # logging.warning(records[exp_name].keys())
                    all_log_probs.append(log_probs)
                # \'''
                all_log_probs = torch.stack(all_log_probs, dim=0)  # 3 * N * T * C
                ens_algo = "dtw"
                for _idx, _seqs in enumerate(zip(*all_log_probs)):
                    
                    merged_seqs = self.merge_all(list(_seqs), algo=ens_algo)  # 3 * T * C
                    # all_log_probs[:, _idx, :, :] = torch.stack(merged_seqs, dim=0)
                    if ens_algo == "dtw":
                        self.real_downstream.model(
                            split,
                            features, *others,
                            records = records["real"],
                            batch_id = batch_id,
                            log_probs = torch.stack(merged_seqs).mean(dim=0).unsqueeze(0)
                        )
                    else:
                        num_classes = all_log_probs.shape[-1]
                        self.real_downstream.model(
                            split,
                            features, *others,
                            records = records["real"],
                            batch_id = batch_id,
                            log_probs = torch.nn.functional.one_hot(merged_seqs, num_classes).unsqueeze(0).float()
                        )
                # \'''
                batch_ids.append(batch_id)
                \'''
                avg_log_probs = torch.stack(all_log_probs, dim=0).mean(dim=0)
                
                self.real_downstream.model(
                    split,
                    features, *others,
                    records = records["real"],
                    batch_id = batch_id,
                    log_probs = avg_log_probs
                )
                \'''
                
        # \'''
        '''
        # logging
        for _exp_dir, _record in records.items(): 
            logging.warning(_record.keys())
            if _exp_dir == "real":
                continue
            hyp_cons = _record["hyp_consecutive"]
            gts = _record["groundtruth"]
            
            logging.warning(_exp_dir)
            with open(os.path.join(_exp_dir, 'hyp_cons.json'), 'w') as f:
                json_data = json.dumps(hyp_cons, cls=NpEncoder)
                f.write(json_data)
            with open(os.path.join(_exp_dir, 'gts.json'), 'w') as f:
                json_data = json.dumps(gts, cls=NpEncoder)
                f.write(json_data)

        save_names = self.real_downstream.model.log_records(
            split,
            records = records["real"],
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

        # for entry, training in zip(self.all_entries, trainings):
        #     if training:
        #         entry.model.train()

        if not_during_training:
            logger.close()
            shutil.rmtree(tempdir)
        linelogger.info(save_names)
        return [] if type(save_names) is not list else save_names
    
    # @torch.jit.script
 
        # '''
    
        # waited_futures = []
        # for _fut in futures_:
            # waited_futures.extend(torch.jit.wait(_fut))
            
        for _idx, merged_seqs in enumerate(waited_futures):
            if ens_algo == "dtw":
                self.real_downstream.model(
                    split,
                    features, *others,
                    records = records["real"],
                    batch_id = batch_id,
                    log_probs = torch.stack(merged_seqs).mean(dim=0).unsqueeze(0)
                )
            else:
                num_classes = all_log_probs.shape[-1]
                self.real_downstream.model(
                    split,
                    features, *others,
                    records = records["real"],
                    batch_id = batch_id,
                    log_probs = torch.nn.functional.one_hot(merged_seqs, num_classes).unsqueeze(0).float()
                )
              
              
    @ray.remote  
    def single_merge(self, batch_object, split, records: Dict, ens_algo: str="dtw") -> List[List[torch.Tensor]]:
        batch_id, (wavs, *others) = batch_object
        wavs = [torch.FloatTensor(wav).to(self.args.device) for wav in wavs]
        all_log_probs: List[torch.Tensor] = []
        logging.warning("Start merging")
        with torch.no_grad():
            for upstream, featurizer, downstream, ens_ckpt in zip(self.upstreams, self.featurizers, self.downstreams, self.args.ensemble):
                features = upstream.model(wavs)
                features = featurizer.model(wavs, features)
                exp_name = os.path.dirname(ens_ckpt)
                log_probs = downstream.model(
                    split,
                    features, *others,
                    records = records[exp_name],
                    batch_id = batch_id,
                    return_log_probs = True
                )
                # logging.warning(records[exp_name].keys())
                all_log_probs.append(log_probs)
            # '''
            all_log_probs: torch.Tensor = torch.stack(all_log_probs, dim=0)  # 3 * N * T * C
                
        all_merged_seqs = []
        for _idx, _seqs in enumerate(zip(*all_log_probs)):
            merged_seqs = iso_merge_all(list(_seqs), algo=ens_algo)  # 3 * T * C
            all_merged_seqs.append(merged_seqs)
            # all_log_probs[:, _idx, :, :] = torch.stack(merged_seqs, dim=0)
        
        return all_merged_seqs
     
                
    # @torch.jit.script
    def parallel_monkey(self, dataloader, split, evaluate_steps, records, batch_ids: List[int]):
        ctx = torch.multiprocessing.get_context('spawn')
        # with ctx.Pool(16) as pool:
        #     results = pool.map(functools.partial(self.single_merge, split=split, records=records), enumerate(dataloader))
        results = [self.single_merge.remote(self, batch_object, split, records) for batch_object in enumerate(dataloader)]
        return ray.get(results)
        with torch.multiprocessing.Pool(16) as pool:
            results = pool.starmap(functools.partial(self.single_merge, split=split, records=records), (enumerate(dataloader,)))
        return results
        futures_ = []
        for batch_id, (wavs, *others) in enumerate(dataloader):
            if batch_id > evaluate_steps:
                break

            wavs = [torch.FloatTensor(wav).to(self.args.device) for wav in wavs]
            all_log_probs: List[torch.Tensor] = []
            with torch.no_grad():
                for upstream, featurizer, downstream, ens_ckpt in zip(self.upstreams, self.featurizers, self.downstreams, self.args.ensemble):
                    features = upstream.model(wavs)
                    features = featurizer.model(wavs, features)
                    exp_name = os.path.dirname(ens_ckpt)
                    log_probs = downstream.model(
                        split,
                        features, *others,
                        records = records[exp_name],
                        batch_id = batch_id,
                        return_log_probs = True
                    )
                    # logging.warning(records[exp_name].keys())
                    all_log_probs.append(log_probs)
                # '''
                all_log_probs: torch.Tensor = torch.stack(all_log_probs, dim=0)  # 3 * N * T * C
                ens_algo = "dtw"
                fut = torch.jit.fork(self.single_merge, all_log_probs, ens_algo)
                # logging.warning(type(fut))
                # futures_.append(fut)
                # batch_ids.append(batch_id)
        
        torch.jit.wait(fut)

    
    def compare_sequence(self, s1, s2, blank1=None, blank2=None, use_monkey=True):
        # T * C
        s1 = s1.detach().cpu().numpy()
        s2 = s2.detach().cpu().numpy()
        t1, t2 = s1.shape[0], s2.shape[0]
        dp = np.zeros([t1, t2]) + np.inf
        frm = [[(-1, -1) for j in range(t2)] for i in range(t1)]
        dp[0][0] = np.linalg.norm(s1[0] - s2[0], ord=2)
        logging.warning(f"{s1.shape}, {s2.shape}")
        if use_monkey:
            for i in range(t1):
                for j in range(t2):
                    if not j and not i:
                        continue
                    if not j:
                        dp[i][j] = dp[i - 1][j] + np.linalg.norm(s1[i] - s2[j], ord=2)
                        frm[i][j] = (i - 1, j)
                    elif not i:
                        dp[i][j] = dp[i][j - 1] + np.linalg.norm(s1[i] - s2[j], ord=2)
                        frm[i][j] = (i, j - 1)
                    else:
                        for di, dj in [(-1, 0), (-1, -1), (0, -1)]:
                            nx, ny = i + di, j + dj
                            if abs(i - j) < 2:
                                pass
                                # logging.warning(f"({nx}, {ny}) : {dp[nx][ny] + np.linalg.norm(s1[i] - s2[j], ord=2):.5f}")
                            if dp[nx][ny] + np.linalg.norm(s1[i] - s2[j], ord=2) < dp[i][j]:
                                frm[i][j] = (nx, ny)
                                dp[i][j] = dp[nx][ny] + np.linalg.norm(s1[i] - s2[j], ord=2)
            
            ret1, ret2 = [], []
            nx, ny = t1 - 1, t2 - 1
            while(nx > 0 or ny > 0):
                ret1.append(nx)
                ret2.append(ny)
                fx, fy = frm[nx][ny]
                # print(f"lattice: {nx}, {ny}, {fx}, {fy}")
                nx, ny = fx, fy
            ret1.append(0)
            ret2.append(0)
        else:
            _dtw = SoftDTW(use_cuda=True)
            _res = _dtw(torch.from_numpy(s1).unsqueeze(0), torch.from_numpy(s2).unsqueeze(0))[0, :-1, :-1]
            ret1, ret2 = [], []
            nx, ny = t1, t2
            while(nx > 0 or ny > 0):
                ret1.append(nx)
                ret2.append(ny)
                _frx, _fry = nx, ny
                _max_dis = torch.inf
                for di, dj in [(-1, 0), (-1, -1), (0, -1)]:
                    fx, fy = nx + di, ny + dj
                    if _res[fx, fy] + torch.norm(s1[nx - 1] - s2[ny - 1], p=2) < _max_dis:
                        _max_dis = _res[fx, fy] + torch.norm(s1[nx - 1] - s2[ny - 1], p=2)
                        _frx, _fry = fx, fy
                        
                # print(f"lattice: {nx}, {ny}, {fx}, {fy}")
                nx, ny = _frx, _fry
            ret1.append(0)
            ret2.append(0)
        # logging.info(ret2)
        
        logging.warning(f"{len(ret1)}, {len(ret2)}")
        ret1 = torch.tensor(ret1[::-1])
        ret2 = torch.tensor(ret2[::-1])
        logging.warning(ret1)
        logging.warning(ret2)
        new_blank1, new_blank2 = [], []
        if blank1 and blank2:
            max_id = -1
            blank_id = 0
            for _i, e in enumerate(ret1):
                if blank_id >= len(blank1):
                    break
                if e > max_id:
                    if e == blank1[blank_id]:
                        new_blank1.append(_i)
                        blank_id += 1
                    max_id = e
            
            max_id = -1
            blank_id = 0
            for _i, e in enumerate(ret2):
                if blank_id >= len(blank2):
                    break
                if e > max_id:
                    if e == blank2[blank_id]:
                        new_blank2.append(_i)
                        blank_id += 1
                    max_id = e
            
            new_blank1 = sorted(list(set(new_blank1)))
            new_blank2 = sorted(list(set(new_blank2)))
            new_blank1 = torch.tensor(new_blank1)
            new_blank2 = torch.tensor(new_blank2)
        logging.warning(f"{new_blank1}")
        logging.warning(f"{new_blank2}")
        return (s1[ret1] + s2[ret2]) / 2, ret1, ret2, dp[-1][-1], new_blank1, new_blank2

    @staticmethod
    def parse_blank_pos(blank):
        new_blank = []
        cnt_ = 0
        for _i, e in enumerate(blank):
            if e:
                cnt_ += 1
            else:
                if not _i or blank[_i - 1]:
                    new_blank.append(cnt_)

        return new_blank

    @staticmethod
    def to_onehot(seq):
        # seq: [T * C] -> T * C, C is the number of classes
        # [0.2, 0.3, 0.5] -> [0, 0, 1]
        # use torch scatter
        top_dim = torch.argmax(seq, dim=-1)
        return torch.nn.functional.one_hot(top_dim, num_classes=seq.shape[-1]).float()
        
    def LCS(self, s1, s2):
        dp = np.zeros([len(s1) + 1, len(s2) + 1]) + np.inf
        frm = [[(-1, -1) for j in range(len(s2) + 1)] for i in range(len(s1) + 1)]
        dp[0][0] = 0
        for i in range(1, len(s1) + 1):
            dp[i][0] = i
            frm[i][0] = (i - 1, 0)
        for j in range(1, len(s2) + 1):
            dp[0][j] = j
            frm[0][j] = (0, j - 1)
        for i in range(1, len(s1) + 1):
            for j in range(1, len(s2) + 1):
                for di, dj in [(-1, 0), (-1, -1), (0, -1)]:
                    nx, ny = i + di, j + dj
                    if dp[nx][ny] + (s1[i - 1] != s2[j - 1]) < dp[i][j]:
                        frm[i][j] = (nx, ny)
                        dp[i][j] = dp[nx][ny] + (s1[i - 1] != s2[j - 1])
                            
        seq = []
        logging.warning(f"Start LCS backtrack")
        nx, ny = len(s1), len(s2)
        while(nx > 0 or ny > 0):
            # logging.warning(f"{nx}, {ny}")
            fx, fy = frm[nx][ny]
            if fx == nx - 1 and fy == ny - 1:
                seq.append(s1[nx - 1])
            nx, ny = fx, fy
            
        return torch.tensor(seq[::-1])
    
    def merge_all(self, seqs, blank_prob=-1, algo=" ") -> List[torch.Tensor]:
        processed_idx = 1
        blanks = [[] for _ in range(len(seqs))]
        for _sid, _seq in enumerate(seqs):
            soft_seq = torch.nn.functional.softmax(_seq, dim=-1)
            log_seq = torch.nn.functional.log_softmax(_seq, dim=-1)
            if algo == "rover":
                soft_seq = torch.argmax(soft_seq, dim=-1)
            if blank_prob >= 0:
                black_probs = soft_seq[:, self.real_downstream.model.tokenizer.pad_idx]
                logging.warning(f"{torch.sort(black_probs, descending=True)[0][:10]}")
                # filter idx that blank prob > 0.97
                high_prob_blank_idx = (soft_seq[:, self.real_downstream.model.tokenizer.pad_idx] < blank_prob)
                logging.warning(f"{torch.sum(~high_prob_blank_idx)}")
                blanks[_sid] = EnsembleRunner.parse_blank_pos(high_prob_blank_idx)
                logging.warning(f"{blanks[_sid]}")
                seqs[_sid] = soft_seq[high_prob_blank_idx]
            else:
                seqs[_sid] = soft_seq
        copied_seqs = [seqs[0]]
        copied_blanks = [blanks[0]]
        while ((processed_idx) < len(seqs)):
            logging.warn(f"processing {processed_idx}")
            if algo == "dtw":
                aligned_seq, aln1, aln2, score, *ret = self.compare_sequence(copied_seqs[-1], seqs[processed_idx], copied_blanks[-1], blanks[processed_idx])
                for i, _seq in enumerate(copied_seqs):
                    copied_seqs[i] = _seq[aln1]
                    copied_blanks[i] = ret[0]
                copied_seqs.append(seqs[processed_idx][aln2])
                copied_blanks.append(ret[0])
                processed_idx += 1
            else:
                # LCS
                logging.warning("LCS")
                aligned_seq = self.LCS(copied_seqs[-1], seqs[processed_idx])
                copied_seqs.append(aligned_seq)
                processed_idx += 1
        if algo == "rover":
            return copied_seqs[-1]
            

        logging.warning(copied_blanks[-1])
        if blank_prob >= 0:
            for _i, _seq in enumerate(copied_seqs):
                new_seq = []
                _blank_id = 0
                for _j, _token in enumerate(_seq):
                    if(_blank_id < len(copied_blanks[-1])):
                        pass
                        # logging.warning(f"Dbg: {_j}, {_blank_id}, {copied_blanks[-1][_blank_id]}")
                    if(_blank_id < len(copied_blanks[-1]) and copied_blanks[-1][_blank_id] <= _j):
                        _blank_id += 1
                        # append a one hot tensor with pad_idx
                        new_seq.append(torch.zeros_like(_token))
                        new_seq[-1][self.real_downstream.model.tokenizer.pad_idx] = 1.
                    new_seq.append(_token)
                
                copied_seqs[_i] = torch.stack(new_seq)
                logging.warning(torch.where(copied_seqs[_i][:, self.real_downstream.model.tokenizer.pad_idx] > blank_prob))
        return copied_seqs
            
            
            
# MIT License
#
# Copyright (c) 2020 Mehran Maghoumi
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
import torch
import torch.cuda
from numba import jit, prange
from torch.autograd import Function
from numba import cuda
import math

# ----------------------------------------------------------------------------------------------------------------------
@cuda.jit
def compute_softdtw_cuda(D, gamma, bandwidth, max_i, max_j, n_passes, R):
    """
    :param seq_len: The length of the sequence (both inputs are assumed to be of the same size)
    :param n_passes: 2 * seq_len - 1 (The number of anti-diagonals)
    """
    # Each block processes one pair of examples
    b = cuda.blockIdx.x
    # We have as many threads as seq_len, because the most number of threads we need
    # is equal to the number of elements on the largest anti-diagonal
    tid = cuda.threadIdx.x

    # Compute I, J, the indices from [0, seq_len)

    # The row index is always the same as tid
    I = tid

    inv_gamma = 1.0 / gamma

    # Go over each anti-diagonal. Only process threads that fall on the current on the anti-diagonal
    for p in range(n_passes):

        # The index is actually 'p - tid' but need to force it in-bounds
        J = max(0, min(p - tid, max_j - 1))

        # For simplicity, we define i, j which start from 1 (offset from I, J)
        i = I + 1
        j = J + 1

        # Only compute if element[i, j] is on the current anti-diagonal, and also is within bounds
        if I + J == p and (I < max_i and J < max_j):
            # Don't compute if outside bandwidth
            if not (abs(i - j) > bandwidth > 0):
                r0 = -R[b, i - 1, j - 1] * inv_gamma
                r1 = -R[b, i - 1, j] * inv_gamma
                r2 = -R[b, i, j - 1] * inv_gamma
                rmax = max(max(r0, r1), r2)
                rsum = math.exp(r0 - rmax) + math.exp(r1 - rmax) + math.exp(r2 - rmax)
                softmin = -gamma * (math.log(rsum) + rmax)
                R[b, i, j] = D[b, i - 1, j - 1] + softmin

        # Wait for other threads in this block
        cuda.syncthreads()

# ----------------------------------------------------------------------------------------------------------------------
@cuda.jit
def compute_softdtw_backward_cuda(D, R, inv_gamma, bandwidth, max_i, max_j, n_passes, E):
    k = cuda.blockIdx.x
    tid = cuda.threadIdx.x

    # Indexing logic is the same as above, however, the anti-diagonal needs to
    # progress backwards
    I = tid

    for p in range(n_passes):
        # Reverse the order to make the loop go backward
        rev_p = n_passes - p - 1

        # convert tid to I, J, then i, j
        J = max(0, min(rev_p - tid, max_j - 1))

        i = I + 1
        j = J + 1

        # Only compute if element[i, j] is on the current anti-diagonal, and also is within bounds
        if I + J == rev_p and (I < max_i and J < max_j):

            if math.isinf(R[k, i, j]):
                R[k, i, j] = -math.inf

            # Don't compute if outside bandwidth
            if not (abs(i - j) > bandwidth > 0):
                a = math.exp((R[k, i + 1, j] - R[k, i, j] - D[k, i + 1, j]) * inv_gamma)
                b = math.exp((R[k, i, j + 1] - R[k, i, j] - D[k, i, j + 1]) * inv_gamma)
                c = math.exp((R[k, i + 1, j + 1] - R[k, i, j] - D[k, i + 1, j + 1]) * inv_gamma)
                E[k, i, j] = E[k, i + 1, j] * a + E[k, i, j + 1] * b + E[k, i + 1, j + 1] * c

        # Wait for other threads in this block
        cuda.syncthreads()

# ----------------------------------------------------------------------------------------------------------------------
class _SoftDTWCUDA(Function):
    """
    CUDA implementation is inspired by the diagonal one proposed in https://ieeexplore.ieee.org/document/8400444:
    "Developing a pattern discovery method in time series data and its GPU acceleration"
    """

    @staticmethod
    def forward(ctx, D, gamma, bandwidth):
        dev = D.device
        dtype = D.dtype
        gamma = torch.cuda.FloatTensor([gamma])
        bandwidth = torch.cuda.FloatTensor([bandwidth])

        B = D.shape[0]
        N = D.shape[1]
        M = D.shape[2]
        threads_per_block = max(N, M)
        n_passes = 2 * threads_per_block - 1

        # Prepare the output array
        R = torch.ones((B, N + 2, M + 2), device=dev, dtype=dtype) * math.inf
        R[:, 0, 0] = 0

        # Run the CUDA kernel.
        # Set CUDA's grid size to be equal to the batch size (every CUDA block processes one sample pair)
        # Set the CUDA block size to be equal to the length of the longer sequence (equal to the size of the largest diagonal)
        compute_softdtw_cuda[B, threads_per_block](cuda.as_cuda_array(D.detach()),
                                                   gamma.item(), bandwidth.item(), N, M, n_passes,
                                                   cuda.as_cuda_array(R))
        ctx.save_for_backward(D, R.clone(), gamma, bandwidth)
        return R
        return R[:, -2, -2]

    @staticmethod
    def backward(ctx, grad_output):
        dev = grad_output.device
        dtype = grad_output.dtype
        D, R, gamma, bandwidth = ctx.saved_tensors

        B = D.shape[0]
        N = D.shape[1]
        M = D.shape[2]
        threads_per_block = max(N, M)
        n_passes = 2 * threads_per_block - 1

        D_ = torch.zeros((B, N + 2, M + 2), dtype=dtype, device=dev)
        D_[:, 1:N + 1, 1:M + 1] = D

        R[:, :, -1] = -math.inf
        R[:, -1, :] = -math.inf
        R[:, -1, -1] = R[:, -2, -2]

        E = torch.zeros((B, N + 2, M + 2), dtype=dtype, device=dev)
        E[:, -1, -1] = 1

        # Grid and block sizes are set same as done above for the forward() call
        compute_softdtw_backward_cuda[B, threads_per_block](cuda.as_cuda_array(D_),
                                                            cuda.as_cuda_array(R),
                                                            1.0 / gamma.item(), bandwidth.item(), N, M, n_passes,
                                                            cuda.as_cuda_array(E))
        E = E[:, 1:N + 1, 1:M + 1]
        return grad_output.view(-1, 1, 1).expand_as(E) * E, None, None


# ----------------------------------------------------------------------------------------------------------------------
#
# The following is the CPU implementation based on https://github.com/Sleepwalking/pytorch-softdtw
# Credit goes to Kanru Hua.
# I've added support for batching and pruning.
#
# ----------------------------------------------------------------------------------------------------------------------
@jit(nopython=True, parallel=True)
def compute_softdtw(D, gamma, bandwidth):
    B = D.shape[0]
    N = D.shape[1]
    M = D.shape[2]
    R = np.ones((B, N + 2, M + 2)) * np.inf
    R[:, 0, 0] = 0
    for b in prange(B):
        for j in range(1, M + 1):
            for i in range(1, N + 1):

                # Check the pruning condition
                if 0 < bandwidth < np.abs(i - j):
                    continue

                r0 = -R[b, i - 1, j - 1] / gamma
                r1 = -R[b, i - 1, j] / gamma
                r2 = -R[b, i, j - 1] / gamma
                rmax = max(max(r0, r1), r2)
                rsum = np.exp(r0 - rmax) + np.exp(r1 - rmax) + np.exp(r2 - rmax)
                softmin = - gamma * (np.log(rsum) + rmax)
                R[b, i, j] = D[b, i - 1, j - 1] + softmin
    return R

# ----------------------------------------------------------------------------------------------------------------------
@jit(nopython=True, parallel=True)
def compute_softdtw_backward(D_, R, gamma, bandwidth):
    B = D_.shape[0]
    N = D_.shape[1]
    M = D_.shape[2]
    D = np.zeros((B, N + 2, M + 2))
    E = np.zeros((B, N + 2, M + 2))
    D[:, 1:N + 1, 1:M + 1] = D_
    E[:, -1, -1] = 1
    R[:, :, -1] = -np.inf
    R[:, -1, :] = -np.inf
    R[:, -1, -1] = R[:, -2, -2]
    for k in prange(B):
        for j in range(M, 0, -1):
            for i in range(N, 0, -1):

                if np.isinf(R[k, i, j]):
                    R[k, i, j] = -np.inf

                # Check the pruning condition
                if 0 < bandwidth < np.abs(i - j):
                    continue

                a0 = (R[k, i + 1, j] - R[k, i, j] - D[k, i + 1, j]) / gamma
                b0 = (R[k, i, j + 1] - R[k, i, j] - D[k, i, j + 1]) / gamma
                c0 = (R[k, i + 1, j + 1] - R[k, i, j] - D[k, i + 1, j + 1]) / gamma
                a = np.exp(a0)
                b = np.exp(b0)
                c = np.exp(c0)
                E[k, i, j] = E[k, i + 1, j] * a + E[k, i, j + 1] * b + E[k, i + 1, j + 1] * c
    return E[:, 1:N + 1, 1:M + 1]

# ----------------------------------------------------------------------------------------------------------------------
class _SoftDTW(Function):
    """
    CPU implementation based on https://github.com/Sleepwalking/pytorch-softdtw
    """

    @staticmethod
    def forward(ctx, D, gamma, bandwidth):
        dev = D.device
        dtype = D.dtype
        gamma = torch.Tensor([gamma]).to(dev).type(dtype)  # dtype fixed
        bandwidth = torch.Tensor([bandwidth]).to(dev).type(dtype)
        D_ = D.detach().cpu().numpy()
        g_ = gamma.item()
        b_ = bandwidth.item()
        R = torch.Tensor(compute_softdtw(D_, g_, b_)).to(dev).type(dtype)
        ctx.save_for_backward(D, R, gamma, bandwidth)
        return R
        return R[:, -2, -2]

    @staticmethod
    def backward(ctx, grad_output):
        dev = grad_output.device
        dtype = grad_output.dtype
        D, R, gamma, bandwidth = ctx.saved_tensors
        D_ = D.detach().cpu().numpy()
        R_ = R.detach().cpu().numpy()
        g_ = gamma.item()
        b_ = bandwidth.item()
        E = torch.Tensor(compute_softdtw_backward(D_, R_, g_, b_)).to(dev).type(dtype)
        return grad_output.view(-1, 1, 1).expand_as(E) * E, None, None

# ----------------------------------------------------------------------------------------------------------------------
class SoftDTW(torch.nn.Module):
    """
    The soft DTW implementation that optionally supports CUDA
    """

    def __init__(self, use_cuda, gamma=1.0, normalize=False, bandwidth=None, dist_func=None):
        """
        Initializes a new instance using the supplied parameters
        :param use_cuda: Flag indicating whether the CUDA implementation should be used
        :param gamma: sDTW's gamma parameter
        :param normalize: Flag indicating whether to perform normalization
                          (as discussed in https://github.com/mblondel/soft-dtw/issues/10#issuecomment-383564790)
        :param bandwidth: Sakoe-Chiba bandwidth for pruning. Passing 'None' will disable pruning.
        :param dist_func: Optional point-wise distance function to use. If 'None', then a default Euclidean distance function will be used.
        """
        super(SoftDTW, self).__init__()
        self.normalize = normalize
        self.gamma = gamma
        self.bandwidth = 0 if bandwidth is None else float(bandwidth)
        self.use_cuda = use_cuda

        # Set the distance function
        if dist_func is not None:
            self.dist_func = dist_func
        else:
            self.dist_func = SoftDTW._euclidean_dist_func

    def _get_func_dtw(self, x, y):
        """
        Checks the inputs and selects the proper implementation to use.
        """
        bx, lx, dx = x.shape
        by, ly, dy = y.shape
        # Make sure the dimensions match
        assert bx == by  # Equal batch sizes
        assert dx == dy  # Equal feature dimensions

        use_cuda = self.use_cuda

        if use_cuda and (lx > 1024 or ly > 1024):  # We should be able to spawn enough threads in CUDA
                print("SoftDTW: Cannot use CUDA because the sequence length > 1024 (the maximum block size supported by CUDA)")
                use_cuda = False

        # Finally, return the correct function
        return _SoftDTWCUDA.apply if use_cuda else _SoftDTW.apply

    @staticmethod
    def _euclidean_dist_func(x, y):
        """
        Calculates the Euclidean distance between each element in x and y per timestep
        """
        n = x.size(1)
        m = y.size(1)
        d = x.size(2)
        x = x.unsqueeze(2).expand(-1, n, m, d)
        y = y.unsqueeze(1).expand(-1, n, m, d)
        return torch.pow(x - y, 2).sum(3)

    def forward(self, X, Y):
        """
        Compute the soft-DTW value between X and Y
        :param X: One batch of examples, batch_size x seq_len x dims
        :param Y: The other batch of examples, batch_size x seq_len x dims
        :return: The computed results
        """

        # Check the inputs and get the correct implementation
        func_dtw = self._get_func_dtw(X, Y)

        if self.normalize:
            # Stack everything up and run
            x = torch.cat([X, X, Y])
            y = torch.cat([Y, X, Y])
            D = self.dist_func(x, y)
            out = func_dtw(D, self.gamma, self.bandwidth)
            out_xy, out_xx, out_yy = torch.split(out, X.shape[0])
            return out_xy - 1 / 2 * (out_xx + out_yy)
        else:
            D_xy = self.dist_func(X, Y)
            return func_dtw(D_xy, self.gamma, self.bandwidth)
        
    

# ----------------------------------------------------------------------------------------------------------------------
def timed_run(a, b, sdtw):
    """
    Runs a and b through sdtw, and times the forward and backward passes.
    Assumes that a requires gradients.
    :return: timing, forward result, backward result
    """
    from timeit import default_timer as timer

    # Forward pass
    start = timer()
    forward = sdtw(a, b)
    end = timer()
    t = end - start

    grad_outputs = torch.ones_like(forward)

    # Backward
    start = timer()
    grads = torch.autograd.grad(forward, a, grad_outputs=grad_outputs)[0]
    end = timer()

    # Total time
    t += end - start

    return t, forward, grads

# ----------------------------------------------------------------------------------------------------------------------
def profile(batch_size, seq_len_a, seq_len_b, dims, tol_backward):
    sdtw = SoftDTW(False, gamma=1.0, normalize=False)
    sdtw_cuda = SoftDTW(True, gamma=1.0, normalize=False)
    n_iters = 6

    print("Profiling forward() + backward() times for batch_size={}, seq_len_a={}, seq_len_b={}, dims={}...".format(batch_size, seq_len_a, seq_len_b, dims))

    times_cpu = []
    times_gpu = []

    for i in range(n_iters):
        a_cpu = torch.rand((batch_size, seq_len_a, dims), requires_grad=True)
        b_cpu = torch.rand((batch_size, seq_len_b, dims))
        a_gpu = a_cpu.cuda()
        b_gpu = b_cpu.cuda()

        # GPU
        t_gpu, forward_gpu, backward_gpu = timed_run(a_gpu, b_gpu, sdtw_cuda)

        # CPU
        t_cpu, forward_cpu, backward_cpu = timed_run(a_cpu, b_cpu, sdtw)

        # Verify the results
        assert torch.allclose(forward_cpu, forward_gpu.cpu())
        assert torch.allclose(backward_cpu, backward_gpu.cpu(), atol=tol_backward)

        if i > 0:  # Ignore the first time we run, in case this is a cold start (because timings are off at a cold start of the script)
            times_cpu += [t_cpu]
            times_gpu += [t_gpu]

    # Average and log
    avg_cpu = np.mean(times_cpu)
    avg_gpu = np.mean(times_gpu)
    print("  CPU:     ", avg_cpu)
    print("  GPU:     ", avg_gpu)
    print("  Speedup: ", avg_cpu / avg_gpu)
    print()




class DtwDataset(torch.utils.data.Dataset):
    def __init__(self, loader, upstreams, downstreams, featureizers, real_downstream, split, records, args):
        self.args = args
        self.loader = loader
        self.upstreams = upstreams
        self.downstreams = downstreams
        self.featurizers = featureizers
        self.real_downstream = real_downstream
        self.split = split
        self.ens_algo = "dtw"
        self.batch_ids = []
        self.wavs = []
        self.others = []
        self.records = records
        self.stacked_log_probs = []
        for batch_id, (wavs, *others) in enumerate(loader):
            logging.warning(f"others: {(others)}, {len(others[0][0])}, {len(others[1][0])}")
            self.batch_ids.append(batch_id)
            self.wavs.extend(wavs)
            self.others.append(others)
            wavs = [torch.FloatTensor(wav).to(self.args.device) for wav in wavs]
            all_log_probs = []
            with torch.no_grad():
                for upstream, featurizer, downstream, ens_ckpt in zip(self.upstreams, self.featurizers, self.downstreams, self.args.ensemble):
                    features = upstream.model(wavs)
                    features = featurizer.model(wavs, features)
                    exp_name = os.path.dirname(ens_ckpt)
                    log_probs = downstream.model(
                        split,
                        features, *others,
                        records = records[exp_name],
                        batch_id = batch_id,
                        return_log_probs = True
                    )
                    # logging.warning(records[exp_name].keys())
                    all_log_probs.append(log_probs)
                # \'''
                all_log_probs = torch.stack(all_log_probs, dim=0)  # 3 * N * T * C
                self.stacked_log_probs.extend(all_log_probs.transpose(0, 1).cpu())
        # [(2*b), (2*b)] -> [2 * B]
        self.others = list(zip(*self.others))
        # [(b,b,b,b), (b,b,b,b)]
        self.others = [list(itertools.chain.from_iterable(_other)) for _other in self.others]
        # self.others = [np.concatenate(_other) for _other in self.others]
        logging.warning(f"Len: {len(self.stacked_log_probs)} | {len(self.wavs)} | {len(self.others)}")

    @staticmethod
    def merge_all(seqs, blank_prob=-1, algo=" ", pad_idx=-1) -> List[torch.Tensor]:
        processed_idx = 1
        blanks = [[] for _ in range(len(seqs))]
        for _sid, _seq in enumerate(seqs):
            soft_seq = torch.nn.functional.softmax(_seq, dim=-1)
            log_seq = torch.nn.functional.log_softmax(_seq, dim=-1)
            if algo == "rover":
                soft_seq = torch.argmax(soft_seq, dim=-1)
            if blank_prob >= 0:
                black_probs = soft_seq[:, pad_idx]
                logging.warning(f"{torch.sort(black_probs, descending=True)[0][:10]}")
                # filter idx that blank prob > 0.97
                high_prob_blank_idx = (soft_seq[:, pad_idx] < blank_prob)
                logging.warning(f"{torch.sum(~high_prob_blank_idx)}")
                blanks[_sid] = EnsembleRunner.parse_blank_pos(high_prob_blank_idx)
                logging.warning(f"{blanks[_sid]}")
                seqs[_sid] = soft_seq[high_prob_blank_idx]
            else:
                seqs[_sid] = soft_seq
        copied_seqs = [seqs[0]]
        copied_blanks = [blanks[0]]
        while ((processed_idx) < len(seqs)):
            logging.warn(f"processing {processed_idx}")
            if algo == "dtw":
                aligned_seq, aln1, aln2, score, *ret = DtwDataset.compare_sequence(copied_seqs[-1], seqs[processed_idx], copied_blanks[-1], blanks[processed_idx])
                for i, _seq in enumerate(copied_seqs):
                    copied_seqs[i] = _seq[aln1]
                    copied_blanks[i] = ret[0]
                copied_seqs.append(seqs[processed_idx][aln2])
                copied_blanks.append(ret[0])
                processed_idx += 1
            else:
                # LCS
                logging.warning("LCS")
                aligned_seq = DtwDataset.LCS(copied_seqs[-1], seqs[processed_idx])
                copied_seqs.append(aligned_seq)
                processed_idx += 1
        if algo == "rover":
            return copied_seqs[-1]
            

        logging.warning(copied_blanks[-1])
        if blank_prob >= 0:
            for _i, _seq in enumerate(copied_seqs):
                new_seq = []
                _blank_id = 0
                for _j, _token in enumerate(_seq):
                    if(_blank_id < len(copied_blanks[-1])):
                        pass
                        # logging.warning(f"Dbg: {_j}, {_blank_id}, {copied_blanks[-1][_blank_id]}")
                    if(_blank_id < len(copied_blanks[-1]) and copied_blanks[-1][_blank_id] <= _j):
                        _blank_id += 1
                        # append a one hot tensor with pad_idx
                        new_seq.append(torch.zeros_like(_token))
                        new_seq[-1][pad_idx] = 1.
                    new_seq.append(_token)
                
                copied_seqs[_i] = torch.stack(new_seq)
                logging.warning(torch.where(copied_seqs[_i][:, pad_idx] > blank_prob))
        return copied_seqs
    
    @staticmethod
    def compare_sequence(s1, s2, blank1=None, blank2=None, use_monkey=True):
        # T * C
        s1 = s1.detach().cpu().numpy()
        s2 = s2.detach().cpu().numpy()
        t1, t2 = s1.shape[0], s2.shape[0]
        dp = np.zeros([t1, t2]) + np.inf
        frm = [[(-1, -1) for j in range(t2)] for i in range(t1)]
        dp[0][0] = np.linalg.norm(s1[0] - s2[0], ord=2)
        logging.warning(f"{s1.shape}, {s2.shape}")
        if use_monkey:
            for i in range(t1):
                for j in range(t2):
                    if not j and not i:
                        continue
                    if not j:
                        dp[i][j] = dp[i - 1][j] + np.linalg.norm(s1[i] - s2[j], ord=2)
                        frm[i][j] = (i - 1, j)
                    elif not i:
                        dp[i][j] = dp[i][j - 1] + np.linalg.norm(s1[i] - s2[j], ord=2)
                        frm[i][j] = (i, j - 1)
                    else:
                        for di, dj in [(-1, 0), (-1, -1), (0, -1)]:
                            nx, ny = i + di, j + dj
                            if abs(i - j) < 2:
                                pass
                                # logging.warning(f"({nx}, {ny}) : {dp[nx][ny] + np.linalg.norm(s1[i] - s2[j], ord=2):.5f}")
                            if dp[nx][ny] + np.linalg.norm(s1[i] - s2[j], ord=2) < dp[i][j]:
                                frm[i][j] = (nx, ny)
                                dp[i][j] = dp[nx][ny] + np.linalg.norm(s1[i] - s2[j], ord=2)
            
            ret1, ret2 = [], []
            nx, ny = t1 - 1, t2 - 1
            while(nx > 0 or ny > 0):
                ret1.append(nx)
                ret2.append(ny)
                fx, fy = frm[nx][ny]
                # print(f"lattice: {nx}, {ny}, {fx}, {fy}")
                nx, ny = fx, fy
            ret1.append(0)
            ret2.append(0)
        else:
            _dtw = SoftDTW(use_cuda=True)
            _res = _dtw(torch.from_numpy(s1).unsqueeze(0), torch.from_numpy(s2).unsqueeze(0))[0, :-1, :-1]
            ret1, ret2 = [], []
            nx, ny = t1, t2
            while(nx > 0 or ny > 0):
                ret1.append(nx)
                ret2.append(ny)
                _frx, _fry = nx, ny
                _max_dis = torch.inf
                for di, dj in [(-1, 0), (-1, -1), (0, -1)]:
                    fx, fy = nx + di, ny + dj
                    if _res[fx, fy] + torch.norm(s1[nx - 1] - s2[ny - 1], p=2) < _max_dis:
                        _max_dis = _res[fx, fy] + torch.norm(s1[nx - 1] - s2[ny - 1], p=2)
                        _frx, _fry = fx, fy
                        
                # print(f"lattice: {nx}, {ny}, {fx}, {fy}")
                nx, ny = _frx, _fry
            ret1.append(0)
            ret2.append(0)
        # logging.info(ret2)
        
        logging.warning(f"{len(ret1)}, {len(ret2)}")
        ret1 = torch.tensor(ret1[::-1])
        ret2 = torch.tensor(ret2[::-1])
        logging.warning(ret1)
        logging.warning(ret2)
        new_blank1, new_blank2 = [], []
        if blank1 and blank2:
            max_id = -1
            blank_id = 0
            for _i, e in enumerate(ret1):
                if blank_id >= len(blank1):
                    break
                if e > max_id:
                    if e == blank1[blank_id]:
                        new_blank1.append(_i)
                        blank_id += 1
                    max_id = e
            
            max_id = -1
            blank_id = 0
            for _i, e in enumerate(ret2):
                if blank_id >= len(blank2):
                    break
                if e > max_id:
                    if e == blank2[blank_id]:
                        new_blank2.append(_i)
                        blank_id += 1
                    max_id = e
            
            new_blank1 = sorted(list(set(new_blank1)))
            new_blank2 = sorted(list(set(new_blank2)))
            new_blank1 = torch.tensor(new_blank1)
            new_blank2 = torch.tensor(new_blank2)
        logging.warning(f"{new_blank1}")
        logging.warning(f"{new_blank2}")
        return (s1[ret1] + s2[ret2]) / 2, ret1, ret2, dp[-1][-1], new_blank1, new_blank2


    def __getitem__(self, index):
        all_log_probs: List[torch.Tensor] = []
        logging.warning("Start merging")
        with torch.no_grad():
            all_log_probs: torch.Tensor = self.stacked_log_probs[index]  # 3 * N * T * C
            
            merged_seqs = DtwDataset.merge_all(list(all_log_probs), algo=self.ens_algo)  # 3 * T * C
                # all_log_probs[:, _idx, :, :] = torch.stack(merged_seqs, dim=0)
            return {"seq": torch.stack(merged_seqs).mean(dim=0), "wav": self.wavs[index], "other": [_other[index] for _other in self.others]}
        
        
    def __len__(self):
        return len(self.wavs)        


