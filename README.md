## CCF-BDCI-Sentiment-Analysis-Baseline

The code is modified from https://github.com/huggingface/pytorch-transformers.  

### 1. Introduction of the task

Please refer to https://www.datafountain.cn/competitions/350

### 2. Download dataset 

Download dataset and unzip from the website https://www.datafountain.cn/competitions/350/datasets. And put all into the ./data folder.

### 3. Preprocess dataset

```shell
cd data
python preprocess.py
cd ..
```

### 4. Run Bert-base

```shell
bash run_bert.py
###combine result 5 fold
python combine.py --model_prefix ./model_bert --out_path ./sub.csv
```

***Note:***

actual length = max_seq_length * split_num

actual batch size = per_gpu_train_batch_size * numbers of gpu

### 5. Run Bert Whole Word Masking

```shell
#Download pytorch model weights in the ./chinese_wwm_ex_bert from https://github.com/ymcui/Chinese-BERT-wwm
bash run_bert_wwm_ext.sh
python combine.py --model_prefix ./model_bert_wwm_ext --out_path ./sub.csv
```

### 6. Run XLNet

```shell
#Download pytorch model weights in the ./chinese_xlnet_mid/ from https://github.com/ymcui/Chinese-PreTrained-XLNet
bash run_xlnet.py
python combine.py --model_prefix ./model_bert_wwm_ext --out_path ./sub.csv
```

