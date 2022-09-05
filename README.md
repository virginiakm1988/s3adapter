# s3adapter
## Setting up
1. directly clone this repo
2. cd ./s3adapter/s3prl
```
pip install -e .
```
4. pip install necessary packages
```
pip install huggingface-hub
pip install editdistance
pip install loralib
pip install tensorboardX
```
5. You should be able to run s3prl with adapter version!


## Starting adapter-based tuning

After modifying the three files, you can start adapter-based tuning with the following command line:

    run_downstream.py --adapter houlsby -u hubert -d asr -m train -f -n hubert_asr_houlsby 

make sure the last sys.argv contains the name of adapter, and -f is set to True

The supported adapters includes
*   `adapter` for AdapterBias (https://arxiv.org/abs/2205.00305)
* `houlsby` for Houlsby adapters (https://arxiv.org/abs/1902.00751)
* `bitfit` for Bitfit method (https://arxiv.org/abs/2106.10199)
* `lora` for LoRA (https://github.com/microsoft/LoRA)


Simply tune the whole upstream model using the last hidden representation:

    run_downstream.py  -u hubert -d asr -m train -f -l -1 -n hubert_asr_finetune
