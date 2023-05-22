bash LibriSpeech2/prepare.sh /data1/b08902047/librispeech_raw/ /data1/b08902047/librispeech_final
python3 preprocess/preprocess_libri.py --data_path=/data1/b08902047/librispeech_raw/LibriSpeech/ --output_path=/data1/b08902047/LibriSpeech/dev-other
python3 preprocess/generate_len_for_bucket.py --input_data=/data1/b08902047/librispeech_raw/LibriSpeech/ --output_path=./data

python run_downstream.py --adapter=houlsby -u hubert -d asr -m evaluate -t test-clean --device=cuda:0 -f -n hubert_asr_houlsby_{ops}

ops: sequential, skip, parallel, ex: ops = 2 only considers first 2 paths.
python run_downstream.py --adapter=houlsby -u hubert -d ctc -m train -f -n hubert_ctc -uac upstream/adapterConfig.yaml -c downstream/ctc/libriphone.yaml

** create new container
sudo apt update && sudo apt install tmux -y
sudo ln -s /lib/x86_64-linux-gnu/libtic.so.6.2 /lib/x86_64-linux-gnu/libtinfow.so.6
tmux
conda activate s3adapter


** Stage 1 & Stage 2 **
ngpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

python -m torch.distributed.launch --nproc_per_node $ngpus run_downstream.py --adapter=houlsby -u hubert -d ctc -m train -f -n {exp dir name} -uac upstream/adapterConfig.yaml -c downstream/ctc/libriphone.yaml --ngpu ${ngpus} --online -o "config.runner.gradient_accumulate_steps={original steps / 2}" --stage1_ratio=0.5

SD:
python -m torch.distributed.launch --nproc_per_node $ngpus run_downstream.py --adapter=houlsby -u hubert -d diarization -m train -f -n hubert_sd_25_flr_tau -uac upstream/adapterConfig.yaml -c downstream/diarization/config.yaml --ngpu ${ngpus} --online -o "config.runner.gradient_accumulate_steps=2,,config.adapter_config.switch.tau.init_value=10,,config.adapter_config.switch.tau.stop_value=0.1" --stage1_ratio=0.25 --f_lr

** Testing **
python3 run_downstream.py -m evaluate -t {testing split} -i {ckpt} -c downstream/ctc/libriphone.yaml --adapter=houlsby -u hubert -d ctc -uac upstream/adapterConfig.yaml -n {exp name}

SD:
python3 run_downstream.py -m evaluate -e result/downstream/hubert_sd_25_flr_tau/best-states-dev.ckpt -c downstream/diarization/config.yaml --adapter=houlsby -u hubert -d diarization -uac upstream/adapterConfig.yaml

./downstream/diarization/score.sh result/downstream/hubert_sd_fullytrained downstream/diarization/data/test

Note: 
* -w : train weighted sum
* --stage2_ckpt: ckpt from stage1, we'll only load switch logits to our Adapter Module
    * this parameter should not exists simultaneously with the --init_ckpt (-i)
* If using only SINGLE GPU, remember to delete the .module in self.upstream.model.module.model.... #(還是我們用一張的時候也開DDP啊數碼寶貝...)
* --f_lr: featurizer lr start from stage 2
* adapterConfig tau types: ['const', 'linear', 'exp']
Override:
--overide "config.runner.gradient_accumulate_steps=2,,config.adapter_config.switch.baseline=2,,config.adapter_config.switch.baseline=[1,1,2,2,2,2,2,2,2,0,0,2]"