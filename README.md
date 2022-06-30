# s3adapter

# Setting up adapters for s3prl 
In order to run s3prl with different adapters, you will need to modify 3 files

    fairseq/fairseq/models/wav2vec/wav2vec2.py
    s3prl_adapter/downstream/runner.py
    s3prl_adapter/run_downstream.py

After modifying the three files, you can start adapter-based tuning with the following command line:

    run_downstream.py --adapter True -u hubert -d asr -m train -f -l -1 -n hubert_asr_adapter ## make sure the last sys.argv contains the keyword

The supported adapters includes
*   `adapter` for AdapterBias (https://arxiv.org/abs/2205.00305)
* `houlsby` for Houlsby adapters (https://arxiv.org/abs/1902.00751)
* `bitfit` for Bitfit method (https://arxiv.org/abs/2106.10199)
* `lora` for LoRA (https://github.com/microsoft/LoRA)


Simply tune the whole upstream model using the last hidden representation:

    run_downstream.py --adapter True -u hubert -d asr -m train -f -l -1 -n hubert_asr_finetune

