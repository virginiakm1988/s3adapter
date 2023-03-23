bash LibriSpeech2/prepare.sh /data1/b08902047/librispeech_raw/ /data1/b08902047/librispeech_final
python3 preprocess/preprocess_libri.py --data_path=/data1/b08902047/librispeech_raw/LibriSpeech/ --output_path=/data1/b08902047/LibriSpeech/dev-other
python3 preprocess/generate_len_for_bucket.py --input_data=/data1/b08902047/librispeech_raw/LibriSpeech/ --output_path=./data

python run_downstream.py --adapter=houlsby -u hubert -d asr -m evaluate -t test-clean --device=cuda:0 -f -n hubert_asr_houlsby_{ops}

ops: sequential, skip, parallel, ex: ops = 2 only considers first 2 paths.
python run_downstream.py --adapter=houlsby -u hubert -d ctc -m train -f -n hubert_ctc -uac upstream/adapterConfig.yaml -c downstream/ctc/libriphone.yaml

ngpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
python -m torch.distributed.launch --nproc_per_node $ngpus run_downstream.py --adapter=houlsby -u hubert -d ctc -m train_stage1 -f -n hubert_ctc_stage1 -uac upstream/adapterConfig.yaml -c downstream/ctc/libriphone.yaml --ngpu ${ngpus}

** Stage 2 **
python -m torch.distributed.launch --nproc_per_node $ngpus run_downstream.py --adapter=houlsby -u hubert -d ctc -m train_stage2 -f -n hubert_ctc_stage2 -uac upstream/adapterConfig.yaml -c downstream/ctc/libriphone.yaml --ngpu ${ngpus} -w --stage2_ckpt result/downstream/hubert_ctc_stage1/dev-best.ckpt

Note: 
* -w : train weighted sum
* --stage2_ckpt: ckpt from stage1, we'll only load switch logits to our Adapter Module
* If using one GPU, remember to delete the .module in self.upstream.model.module.model....
