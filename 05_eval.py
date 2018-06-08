from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
import numpy as np
import os


def preprocess_input(x):
    x /= 127.5
    x -= 1.
    return x


test_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

test_generator = test_gen.flow_from_directory(
    "./data/test", (224, 224), shuffle=False)


models = ['./backup/resnet_model_epoch45.h5']
for model_name in models:
    model = load_model(os.path.join(model_name))
    print("{} --> {}".format(model_name,model.evaluate_generator(test_generator)))
