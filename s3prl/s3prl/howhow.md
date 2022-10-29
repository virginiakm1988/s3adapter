bash LibriSpeech2/prepare.sh /data1/b08902047/librispeech_raw/ /data1/b08902047/librispeech_final
python3 preprocess/preprocess_libri.py --data_path=/data1/b08902047/librispeech_raw/LibriSpeech/ --output_path=/data1/b08902047/LibriSpeech/dev-other
python3 preprocess/generate_len_for_bucket.py --input_data=/data1/b08902047/librispeech_raw/LibriSpeech/ --output_path=./data

python run_downstream.py --adapter=houlsby -u hubert -d asr -m evaluate -t test-clean --device=cuda:0 -f -n hubert_asr_houlsby_{ops}

ops: skip, sequential, parallel, ex: ops = 2 only considers first 2 paths.