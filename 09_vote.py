import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np
import json
from collections import Counter

pred0 = np.load('xception_pred.npy')
pred1 = np.load('densenet_pred.npy')
pred2 = np.load('inception_resnet_pred.npy')
pred3 = np.load('resnet_pred.npy')

lookup_table = json.load(open('lookuptable.json'))
with open('filenames.csv') as f:
    filenames = [line.strip() for line in f.readlines()]

dfs = []
for pred in [pred0, pred1, pred2, pred3]:

    results = pred
    results = list(np.argmax(results, axis=1))

    prediction = [int(lookup_table[str(r)]) for r in results]

    dfs.append(DataFrame({"name": filenames, "label": prediction}))

df_all = dfs[0][['name', 'label']]
for i in range(1,4):
    df_all = pd.merge(df_all, dfs[i], on=['name'])
df_all['label'] = -1


labels = df_all.iloc[:, 1:]


for i in range(1000):
    c = Counter(labels.iloc[i].values)
    df_all.loc[i, 'label'] = c.most_common()[0][0]


result_df = df_all[['name', 'label']]
df_gt = pd.read_csv("/home/huligang/workspace/BaiDuXJD2018/datasets/testV1.txt",
                    names=['name', 'label'], sep=' ')

df_in_common = pd.merge(df_gt, result_df, on=['name', 'label'])
acc = float(df_in_common.shape[0]) / 1000
print("vote acc: {}".format(acc))

result_df.to_csv("vote.csv", index=False, header=False, sep=' ')


