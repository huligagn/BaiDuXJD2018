import os
import shutil
import sys

DATA_PATH = '/home/huligang/workspace/BaiDuXJD2018/data'
TRAIN_PATH = '/home/huligang/workspace/BaiDuXJD2018/data/train/'
VAL_PATH = '/home/huligang/workspace/BaiDuXJD2018/data/val/'
TEST_PATH = '/home/huligang/workspace/BaiDuXJD2018/data/test'

TRAIN_FILE = '/home/huligang/workspace/BaiDuXJD2018/datasets/train.txt'
TEST_FILE = '/home/huligang/workspace/BaiDuXJD2018/datasets/testV1.txt'


def rmrf_mkdir(dirname):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.mkdir(dirname)


rmrf_mkdir(DATA_PATH)
rmrf_mkdir(TRAIN_PATH)
rmrf_mkdir(TEST_PATH)
# rmrf_mkdir(VAL_PATH)

# generate train set
with open(TRAIN_FILE) as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip().split(' ')
        image_name = line[0]
        image_label = line[1]

        CATEGORY_PATH = os.path.join(TRAIN_PATH, image_label)
        if not (os.path.exists(CATEGORY_PATH)):
            os.mkdir(CATEGORY_PATH)

        src = os.path.join(
            '/home/huligang/workspace/BaiDuXJD2018/datasets/train', image_name)
        dst = os.path.join(CATEGORY_PATH, image_name)
        os.symlink(src, dst)


# generate test set
with open(TEST_FILE) as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip().split(' ')
        image_name = line[0]
        image_label = line[1]

        CATEGORY_PATH = os.path.join(TEST_PATH, image_label)
        if not (os.path.exists(CATEGORY_PATH)):
            os.mkdir(CATEGORY_PATH)

        src = os.path.join(
            '/home/huligang/workspace/BaiDuXJD2018/datasets/test', image_name)
        dst = os.path.join(CATEGORY_PATH, image_name)
        os.symlink(src, dst)
