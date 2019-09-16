# CCF-BDCI-Sentiment-Analysis-Baseline

1.从该[开源代码](https://github.com/huggingface/pytorch-transformers)中改写的

2.该模型将文本截成k段，分别输入语言模型，然后顶层用GRU拼接起来。好处在于设置小的max_length和更大的k来降低显存占用，因为显存占用是关于长度平方级增长的，而关于k是线性增长的

| 模型 | 线上F1 |
| :------- | :---------: |
| Bert-base | 80.3 |
| Bert-wwm-ext | 80.5 | 
| XLNet-base | 79.25 | 
| XLNet-mid | 79.6 | 
| XLNet-large |  --  |
| Roberta-mid | 80.5 |
| Roberta-large (max_seq_length=512, split_num=1) | 81.25 |


**注:**

1)实际长度 = max_seq_length * split_num

2)实际batch size 大小= per_gpu_train_batch_size * numbers of gpu

3)上面的结果所使用的是4卡GPU，因此batch size为4。如果只有1卡的话，那么per_gpu_train_batch_size应设为4, max_length设置小一些。

4)如果显存太小，可以设置gradient_accumulation_steps参数，比如gradient_accumulation_steps=2，batch size=4，那么就会运行2次，每次batch size为2，累计梯度后更新，等价于batch size=4，但速度会慢两倍。而且迭代次数也要相应提高两倍，即train_steps设为10000

具体batch size可看运行时的log，如：

09/06/2019 21:03:41 - INFO - __main__ -   ***** Running training *****

09/06/2019 21:03:41 - INFO - __main__ -     Num examples = 5872

09/06/2019 21:03:41 - INFO - __main__ -     Batch size = 4

09/06/2019 21:03:41 - INFO - __main__ -     Num steps = 5000


## 赛题说明

请查看该[网站](https://www.datafountain.cn/competitions/350)了解赛题 

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

## Bert Whole Word Masking 模型
从该网站下载pytorch权重，并解压到chinese_wwm_ex_bert目录下:  https://github.com/ymcui/Chinese-BERT-wwm
```shell
bash run_bert_wwm_ext.sh
python combine.py --model_prefix ./model_bert_wwm_ext --out_path ./sub.csv
```

## XLNet-mid 模型
从该网站下载pytorch权重，并解压到./chinese_xlnet_mid/目录下: https://github.com/ymcui/Chinese-PreTrained-XLNet
```shell
bash run_xlnet.sh
python combine.py --model_prefix ./model_xlnet --out_path ./sub.csv
```

## Roberta-mid 模型
从该网站下载tensorflow版本的权重，并解压到./chinese_roberta/目录下: https://github.com/brightmart/roberta_zh
```shell
mv chinese_roberta/bert_config_middle.json chinese_roberta/config.json
python -u -m pytorch_transformers.convert_tf_checkpoint_to_pytorch --tf_checkpoint_path chinese_roberta/ --bert_config_file chinese_roberta/config.json --pytorch_dump_path chinese_roberta/pytorch_model.bin
bash run_roberta.sh
python combine.py --model_prefix ./model_roberta --out_path ./sub.csv
```

