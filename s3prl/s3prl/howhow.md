### Create new container

```bash
sudo apt update && sudo apt install tmux -y
sudo ln -s /lib/x86_64-linux-gnu/libtic.so.6.3 /lib/x86_64-linux-gnu/libtinfow.so.6
tmux
conda activate s3adapter
```

### Run the code

```bash
# if using DDP
ngpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

python -m torch.distributed.launch --nproc_per_node $ngpus run_downstream.py --adapter=houlsby -u hubert -d ctc -m train -f -n {exp dir name} -uac upstream/adapterConfig.yaml -c downstream/ctc/libriphone.yaml --ngpu ${ngpus} --online -o "config.runner.gradient_accumulate_steps={original_steps / ngpus}" --stage1_ratio=0.5

# otherwise
python run_downstream.py --adapter=houlsby -u hubert -d ctc -m train -f -n {exp_dir_name} -uac upstream/adapterConfig.yaml -c downstream/ctc/libriphone.yaml --online -o "config.adapter_config.type={your_search_space}" --stage1_ratio={steps_for_searching} --search_algo={NAS methods}
```

#### Note

##### Search Space

Currently, the supported adapter modules are listed as below.

| module | link | name |
|-|-|-|
| Sequential Adapter | | `seq`|
| Parallel Adapter | | `para`|
| LoRA | | `lora` |
| BitFit | | `bitfit` |
| LNFit | | `lnfit` |
| Skip connection || `skip` |

We use all adapters by default. If you want to use the subset of the adapter modules, please override the config by `-o "config.adapter_config.type=[{adapter_you want to use}]"`

Also, please note that in `S3Delta`, `skip` cannot be specified in the search space.

##### Run the baseline / Re-train the searched architecture

If you want to run the baseline or re-train the searched architecture, please override the configuration by `-o "config.adapter_config.type=[{adapters being used}],,config.adapter_config.switch.baseline=[path_index of the adapter]"`. Also, when re-training the derived architecture, please specified the seach algorithm used for searching.

The `baseline` can be either `int` or `list(int)`. If you only specified an integer (`int`), all layers will use the same adpater. If you want to used different adapters, please specified the format of `list(int)` for the baseline.

As for `s3delta`, the baselines should be the format of `list(list(int))`. Each layer could have multiple adapters, and if the corresponding list for a layer is empty, it will use `skip` by default.

```bash
# Ex
# all layers use the same adapter (sequential adapter in this example)
-o "config.adapter_config.type=['skip', 'seq', 'para'],,config.adapter_config.switch.baseline=1"

# use different adapters
-o "config.adapter_config.type=['skip', 'seq', 'para'],,config.adapter_config.switch.baseline=[0, 1, 2, 0, 0, 1, 1, 2, 2, 0, 1, 2]"

# S3Delta
-o "config.adapter_config.type=['seq', 'para', 'lora'],,config.adapter_config.switch.baseline=[[0, 1], [], [0, 2], [1], [0, 2], [], [1], [0, 2], [2], [], [1, 2], [1]]"
```

##### Supported NAS methods

Currently, the supported NAS methods are listed as below.

| method | link | method name |
|-|-|-|
| DARTS || `darts` |
| GDAS || `gdas`|
| Fair-DARTS || `fair_darts`|
| Gumbel-DARTS || `gumbel_darts`|
| S3Delta || `s3delta`|

The default configurations for each methods are set as `upstream/search_utils.py`.

If you want to use first-order approximation for DARTS-based methods, please override the config by `-o "config.adapter_config.switch.algo.second_order=False"`.

##### Train weighted_sum
If you want to tune featurizer with switch logits, specify `--f_lr` & `--f_lr_stage=1` when running the code. Also, note that currently we only support `weighted_sum` being trained with **network weights**.

#### Testing

```bash
python3 run_downstream.py -m evaluate -t {testing split} -i {ckpt} -c downstream/ctc/libriphone.yaml --adapter=houlsby -u hubert -d ctc -uac upstream/adapterConfig.yaml -n {exp name}
```

#### Ensemble

```bash
CUDA_VISIBLE_DEVICES=1 python3 run_downstream.py -m evaluate -t test -u wav2vec2 -d ctc -f -n test -o "config.adapter_config.type=['seq', 'para', 'lora']" --ensemble result/downstream/wav2vec2_pr_seq_bak/dev-best.ckpt result/downstream/wav2vec2_pr_para_bak/dev-best.ckpt result/downstream/wav2vec2_pr_para_bak/dev-best.ckpt --adapter=houlsby -c downstream/ctc/libriphone.yaml
```