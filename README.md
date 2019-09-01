# CCF-BDCI-Sentiment-Analysis-Baseline

1.代码从[该开源代码](https://github.com/huggingface/pytorch-transformers)改写

2.线上Bert的成绩为80.3 

## 赛题说明

请从查看该[网站](https://www.datafountain.cn/competitions/350)了解赛题 

## 下载数据集

从该[网站](https://www.datafountain.cn/competitions/350/datasets)中下载数据集, 并解压在./data目录。

## 数据预处理

```shell
cd data
python preprocess.py
cd ..
```

## Bert-base 模型

```shell
bash run_bert.sh
#5 fold取平均
python combine.py --model_prefix ./model_bert --out_path ./sub.csv
```

**注意:**

实际长度 = max_seq_length * split_num

实际batch size 大小= per_gpu_train_batch_size * numbers of gpu

## Bert Whole Word Masking 模型

```shell
#从该网站下载权重，并解压到chinese_wwm_ex_bert目录下:  https://github.com/ymcui/Chinese-BERT-wwm
bash run_bert_wwm_ext.sh
python combine.py --model_prefix ./model_bert_wwm_ext --out_path ./sub.csv
```

## XLNet 模型

```shell
#从该网站下载权重，并解压到./chinese_xlnet_mid/目录下: https://github.com/ymcui/Chinese-PreTrained-XLNet
bash run_xlnet.sh
python combine.py --model_prefix ./model_bert_wwm_ext --out_path ./sub.csv
```

