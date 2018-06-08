from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
import numpy as np


def preprocess_input(x):
    x /= 127.5
    x -= 1.
    return x

test_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_generator = test_gen.flow_from_directory(
    "./data/test", (224, 224), shuffle=False, batch_size=1)

model = load_model("./backup/resnet_model_epoch45.h5")

results = model.predict_generator(test_generator, max_queue_size=10,
                  workers=1, use_multiprocessing=False, verbose=0)

results = list(np.argmax(results, axis=1))


train_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
train_generator = train_gen.flow_from_directory(
    "./data/train", (224, 224), shuffle=False)

lookup_table = train_generator.class_indices
lookup_table = dict((v, k) for k, v in lookup_table.items())

prediction = [lookup_table[r] for r in results]


with open('resnet50_epoch45.csv', 'w') as f:
    for name,label in zip(test_generator.filenames, prediction):
        name = name.split('/')[-1]
        f.write(name + ' ' + label + '\n')

print('result saved --> resnet50_epoch45.csv')
