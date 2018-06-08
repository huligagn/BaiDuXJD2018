import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np
import json


def get_acc(RATIO1, RATIO2, RATIO3):
    pred1 = np.load('resnet_pred.npy')
    pred2 = np.load('xception_pred.npy')
    pred3 = np.load('densenet_pred.npy')
    pred4 = np.load('inception_resnet_pred.npy')
    lookup_table = json.load(open('lookuptable.json'))
    with open('filenames.csv') as f:
        filenames = [line.strip() for line in f.readlines()]

    # merge the result
    results = pred1 * RATIO1 + pred2 * RATIO2 + pred3 * \
        RATIO3 + pred4 * (1 - RATIO1 - RATIO2 - RATIO3)
    results = list(np.argmax(results, axis=1))

    prediction = [int(lookup_table[str(r)]) for r in results]

    df_gt = pd.read_csv("/home/huligang/workspace/BaiDuXJD2018/datasets/testV1.txt",
                        names=['name', 'label'], sep=' ')

    df_pred = DataFrame({"name": filenames, "label": prediction})

    assert df_gt.shape == df_pred.shape

    df_in_common = pd.merge(df_gt, df_pred, on=['name', 'label'])

    acc = float(df_in_common.shape[0]) / 1000
    print("Ratio [{:.2f},{:.2f},{:.2f},{:.2F}] --> {}".format(RATIO1,
                                                              RATIO2, RATIO3, 1 - RATIO1 - RATIO2 - RATIO3, acc))

print("Test all: ")
print("--------------")
get_acc(1, 0, 0)
get_acc(0, 1, 0)
get_acc(0, 0, 1)
get_acc(0, 0, 0)
print("--------------")


##############################
# combination algorithm
##############################

def get_total_count(n, k):
    num = 1
    den = 1
    for i in range(k):
        num *= n
        den *= k
        n -= 1
        k -= 1
    return num / den


def combine(n, m):
    result = []

    total_count = get_total_count(n, m)
    dst = [1 for i in range(m)] + [0 for i in range(n - m)]
    result.append(dst.copy())

    count = 1
    while count < total_count:
        i = 0
        k = 0
        while i <= n - 2:
            if dst[i] > dst[i + 1]:
                dst[i] = 0
                dst[i + 1] = 1
                break
            if dst[i] == 1:
                k += 1
            i += 1
        if dst[0] == 0 and k > 0 and i > 0:
            for j in range(k):
                dst[j] = 1
            for j in range(k, i):
                dst[j] = 0
        result.append(dst.copy())
        count += 1

    return result


def get_ratio(result):
    items = []
    for i in range(len(result)):
        item = []
        for j in range(len(result[i])):
            if result[i][j] == 1:
                item.append(j)
        items.append(item)
    return items


##############################
# find the hyper params
##############################
raw = np.array([0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
num_combinations = get_total_count(10, 3)
combinations = combine(10, 3)
ratios = get_ratio(combinations)

for i in range(int(num_combinations)):
    r1 = ratios[i][0]
    r2 = ratios[i][1] - ratios[i][0]
    r3 = ratios[i][2] - ratios[i][1]
    get_acc(r1 / 10, r2 / 10, r3 / 10)


##############################
# generate the result csv file
##############################

def gen_csv(RATIO1, RATIO2, RATIO3):
    pred1 = np.load('resnet_pred.npy')
    pred2 = np.load('xception_pred.npy')
    pred3 = np.load('densenet_pred.npy')
    pred4 = np.load('inception_resnet_pred.npy')
    lookup_table = json.load(open('lookuptable.json'))
    with open('filenames.csv') as f:
        filenames = [line.strip() for line in f.readlines()]

    # merge the result
    results = pred1 * RATIO1 + pred2 * RATIO2 + pred3 * \
        RATIO3 + pred4 * (1 - RATIO1 - RATIO2 - RATIO3)
    results = list(np.argmax(results, axis=1))

    prediction = [int(lookup_table[str(r)]) for r in results]

    # write to file
    with open('merge_result.csv', 'w') as f:
        for name, label in zip(filenames, prediction):
            name = name.split('/')[-1]
            f.write(name + ' ' + str(label) + '\n')

    print('result saved --> merge_result.csv')

    df_gt = pd.read_csv("/home/huligang/workspace/000/Baidu/datasets/testV1.txt",
                        names=['name', 'label'], sep=' ')

    df_pred = DataFrame({"name": filenames, "label": prediction})

    assert df_gt.shape == df_pred.shape

    df_in_common = pd.merge(df_gt, df_pred, on=['name', 'label'])

    acc = float(df_in_common.shape[0]) / 1000
    print("Ratio [{:.2f},{:.2f},{:.2f},{:.2F}] --> {}".format(RATIO1,
                                                              RATIO2, RATIO3, 1 - RATIO1 - RATIO2 - RATIO3, acc))


gen_csv(0.2, 0.4, 0.2)
