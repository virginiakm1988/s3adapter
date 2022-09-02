# s3adapter
## NEW VERSION
1. directly clone this repo
2. cd fairseq, pip install -e .
3. cd ../s3prl, pip install -e .
4. You should be able to run s3prl with adapter version!


#### Setting up adapters for s3prl 
1. Clone the newest s3prl repo first.
2. pip install editdistance
3. pip install loralib
4. In order to run s3prl with different adapters, you will need to modify 3 files within your installed s3prl_repo

    ```
    fairseq/fairseq/models/wav2vec/wav2vec2.py 
    ```
    > ==> modify your fairseq/fairseq/models/wav2vec/wav2vec2.py   
    ```
     s3prl_adapter/downstream/runner.py
    ```
    > ==> modify your ./s3prl/downstream/runner.py

    ```

    s3prl_adapter/run_downstream.py 

    ```
    > ==> modify your ./s3prl/run_downstream.py
 

After modifying the three files, you can start adapter-based tuning with the following command line:

    run_downstream.py --adapter True -u hubert -d asr -m train -f -l -1 -n hubert_asr_adapter 

make sure the last sys.argv contains the name of adapter

The supported adapters includes
*   `adapter` for AdapterBias (https://arxiv.org/abs/2205.00305)
* `houlsby` for Houlsby adapters (https://arxiv.org/abs/1902.00751)
* `bitfit` for Bitfit method (https://arxiv.org/abs/2106.10199)
* `lora` for LoRA (https://github.com/microsoft/LoRA)


Simply tune the whole upstream model using the last hidden representation:

    run_downstream.py --adapter True -u hubert -d asr -m train -f -l -1 -n hubert_asr_finetune
