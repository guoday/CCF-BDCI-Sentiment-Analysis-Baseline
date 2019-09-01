import pandas as pd
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model_prefix", default=None, type=str, required=True)
parser.add_argument("--out_path", default=None, type=str, required=True)
args = parser.parse_args()

k=5
df=pd.read_csv('data/submit_example.csv')
df['0']=0
df['1']=0
df['2']=0
for i in range(k):
    temp=pd.read_csv('{}{}/sub.csv'.format(args.model_prefix,i))
    df['0']+=temp['label_0']/k
    df['1']+=temp['label_1']/k
    df['2']+=temp['label_2']/k
print(df['0'].mean())
 
df['label']=np.argmax(df[['0','1','2']].values,-1)
df[['id','label']].to_csv(args.out_path,index=False)