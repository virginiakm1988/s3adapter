# Previous train() in Runner.py
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
                        if 'switch' in name:
                            trainable_a_paras.append(param)
                        else:
                            trainable_w_paras.append(param)
                            adapter_param += param.nelement() 
                        param.requires_grad = True
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

        delta_step = 0
        batch_ids = []
        records = defaultdict(list)
        epoch = self.init_ckpt.get('Epoch', {'train': 0, 'switch': 0})
        train_split = self.config['runner'].get("train_dataloader", "train")

        linelogger.info(f'train stage for {self.stage1_steps} steps')
        if len(self.adapter_config.adapter.switch.path) > 1 and not self.adapter_config.adapter.switch.baseline:
            # Do search
            adapterModes = ['switch', 'train'] if self.adapter_config.switch.algo in ['gdas'] else ['train', 'switch']
        else:
            adapterModes = ['train'] 
        
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
                            dataloaders[adapterMode].sampler.set_epoch(epoch[adapterMode])
                except:
                    raise
            else:
                raise

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

        while pbar.n < self.config['runner']['total_steps']:
            if self.stage == 1 and pbar.n >= self.stage_steps_prefix[self.stage - 1]:
                self.prepare_stage(2)
                self.stage = 2
                adapterModes = ['train']
                delta_step = self.stage1_steps * 2 - pbar.n 
                inner_pbar.close()
                linelogger.info("to stage2!")
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
                for adapterMode in adapterModes:
                    if dataloaders[adapterMode]:
                        linelogger.info(f'dataset size of {adapterMode}: {len(dataloaders[adapterMode].dataset)}')
                        linelogger.info(f'data loader size of {adapterMode}: {len(dataloaders[adapterMode])}')
                        linelogger.info(f'dataset # indice of {adapterMode}: {len(dataloaders[adapterMode].dataset.indices)}')
            batch_id = -1 # set to -1 so that the first batch's batch_id will be zero.
            inner_pbar = tqdm(total=len(dataloaders['train']), dynamic_ncols=True, desc=f'train_stage{self.stage}', file=tqdm_file)
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
                            epoch[adapterMode] += 1
                            if hasattr(dataloaders[adapterMode], "sampler") and isinstance(dataloaders[adapterMode].sampler, DistributedSampler):
                                dataloaders[adapterMode].sampler.set_epoch(epoch[adapterMode])
                            
                            # Reopen the pbar
                            if adapterMode == 'train':
                                inner_pbar.close()
                                inner_pbar = tqdm(total=len(dataloaders['train']), dynamic_ncols=True, desc=f'train_stage{self.stage}', file=tqdm_file)
                                batch_id = 0

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
                            # Fair-DARTS loss
                            if adapterMode == 'switch' and self.adapter_config.adapter.switch.fair_darts:
                                if isinstance(self.upstream.model, DDP):
                                    aux_loss = self.upstream.model.module.model.aux_loss() * self.adapter_config.adapter.switch.aux_loss_ratio
                                else:
                                    aux_loss = self.upstream.model.model.aux_loss() * self.adapter_config.adapter.switch.aux_loss_ratio
                                loss += aux_loss
                                records['aux_loss'].append(aux_loss.detach().item())
                            
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
      
        pbar.close()
        inner_pbar.close()

        if self.args.push_to_hf_hub:
            self.push_to_huggingface_hub()
        if is_leader_process():
            logger.close()
            wandb.finish()
