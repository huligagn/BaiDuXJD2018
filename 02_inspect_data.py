import os
import matplotlib.pyplot as plt
import cv2

TRAIN_PATH = './data/train/'

classes = os.listdir(TRAIN_PATH)
class_names = []
class_nums = []
img_size_x = []
img_size_y = []
for c in classes:
    images = os.listdir(os.path.join(TRAIN_PATH, c))
    class_names.append(int(c))
    class_nums.append(len(images))
    for img in images:
        (x, y, _) = cv2.imread(os.path.join(TRAIN_PATH, c, img)).shape
        img_size_x.append(x)
        img_size_y.append(y)

plt.bar(class_names, class_nums)
plt.show()
print("{} images in total".format(sum(class_nums)))
print("{} classes in total".format(len(class_names)))
print("{} images/class in average.".format(sum(class_nums) / len(class_names)))
print("min number: {}; max number: {}".format(min(class_nums), max(class_nums)))

print("mean shape of the images: ({}, {}, 3)".format(
    sum(img_size_x) / len((img_size_x)), sum(img_size_y) / len((img_size_y))))
