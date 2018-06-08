from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
import numpy as np
import os
import pandas as pd
import json

from keras.applications.densenet import preprocess_input as densenet_preprocess_input
# from keras.applications.resnet50 import preprocess_input as resnet_preprocess_input

# WAHTCH OUT!!! you should use different preprocess_input function

def preprocess_input(x):
    x /= 127.5
    x -= 1.
    return x

test_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input)

test_gen_densenet = ImageDataGenerator(
    preprocessing_function=densenet_preprocess_input)

test_gen_resnet = ImageDataGenerator(
    preprocessing_function=preprocess_input)

test_generator1 = test_gen.flow_from_directory(
    "./data/test", (299, 299), shuffle=False, batch_size=1)
test_generator2 = test_gen.flow_from_directory(
    "./data/test", (299, 299), shuffle=False, batch_size=1)
test_generator_densenet = test_gen_densenet.flow_from_directory(
    "./data/test", (299, 299), shuffle=False, batch_size=1)
test_generator_resnet = test_gen_resnet.flow_from_directory(
    "./data/test", (224, 224), shuffle=False, batch_size=1)


model_IR105 = load_model('./backup/inception_resnet_model_epoch45.h5')
model_X20 = load_model('./backup/xception_model_epoch40.h5')
model_D60 = load_model('./backup/densenet_model_60.h5')
model_R50 = load_model('./backup/resnet_model_epoch45.h5')

pred_IR105 = model_IR105.predict_generator(test_generator1, max_queue_size=10,
                                 workers=1, use_multiprocessing=False, verbose=0)

pred_X20 = model_X20.predict_generator(test_generator2, max_queue_size=10,
                                 workers=1, use_multiprocessing=False, verbose=0)

pred_D60 = model_D60.predict_generator(test_generator_densenet, max_queue_size=10,
                                 workers=1, use_multiprocessing=False, verbose=0)
pred_R50 = model_R50.predict_generator(test_generator_resnet, max_queue_size=10,
                                 workers=1, use_multiprocessing=False, verbose=0)

np.save("inception_resnet_pred.npy", pred_IR105)
np.save("xception_pred.npy", pred_X20)
np.save("densenet_pred.npy", pred_D60)
np.save("resnet_pred.npy", pred_R50)

# save the filenames
with open('filenames.csv', 'w') as f:
    for name in test_generator1.filenames:
        f.write(name.split('/')[-1] + '\n')

# make the lookuptable
train_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
train_generator = train_gen.flow_from_directory(
    "./data/train", (224, 224), shuffle=False)

lookup_table = train_generator.class_indices
lookup_table = dict((v, k) for k, v in lookup_table.items())

with open("lookuptable.json", "w") as f:
    json.dump(lookup_table, f)
