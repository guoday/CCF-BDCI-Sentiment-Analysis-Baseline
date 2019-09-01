import pandas as pd
import os
import random
import jieba

# 创建停用词列表
def stopwordslist():
    stopwords = [line.strip() for line in open('stopwords/哈工大停用词表.txt',encoding='UTF-8').readlines()]
    return stopwords
# 创建一个停用词列表
stopwords = stopwordslist()
# 对句子进行中文分词
def seg_depart(sentence):
    # 对文档中的每一行进行中文分词
    sentence_depart = jieba.cut(sentence.strip())
    # 输出结果为outstr
    outstr = ''
    # 去停用词
    for word in sentence_depart:
        if word not in stopwords:
                outstr += word
                outstr += " "
    return outstr



train_df=pd.read_csv("Train_DataSet.csv")
train_label_df=pd.read_csv("Train_DataSet_Label.csv")
test_df=pd.read_csv("Test_DataSet.csv")
train_df=train_df.merge(train_label_df,on='id',how='left')
train_df['label']=train_df['label'].fillna(-1)
train_df=train_df[train_df['label']!=-1]
train_df['label']=train_df['label'].astype(int)
test_df['label']=0

test_df['content']=test_df['content'].fillna('无')
train_df['content']=train_df['content'].fillna('无')
test_df['title']=test_df['title'].fillna('无')
train_df['title']=train_df['title'].fillna('无')

index=set(range(train_df.shape[0]))
K_fold=[]
for i in range(5):
    if i == 4:
        tmp=index
    else:
        tmp=random.sample(index,int(1.0/5*train_df.shape[0]))
    index=index-set(tmp)
    print("Number:",len(tmp))
    K_fold.append(tmp)
    

for i in range(5):
    print("Fold",i)
    os.system("mkdir data_{}".format(i))
    dev_index=list(K_fold[i])
    train_index=[]
    for j in range(5):
        if j!=i:
            train_index+=K_fold[j]
    train_df.iloc[train_index].to_csv("data_{}/train.csv".format(i))
    train_df.iloc[dev_index].to_csv("data_{}/dev.csv".format(i))
    test_df.to_csv("data_{}/test.csv".format(i))