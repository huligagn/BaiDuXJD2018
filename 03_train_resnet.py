from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.layers import *
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD


import os

os.mkdir("./model")


def preprocess_input(x):
    x /= 127.5
    x -= 1.
    return x

# create the base pre-trained model
base_model = ResNet50(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(100)(x)
x = BatchNormalization()(x)
predictions = Softmax()(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)


##########################
# data generator
##########################
train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

test_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_generator = train_gen.flow_from_directory(
    "./data/train", (224, 224))

test_generator = test_gen.flow_from_directory(
    "./data/test", (224, 224), shuffle=False)

##########################
# training process
##########################

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
for epoch_num in [5]:
    model.fit_generator(
        train_generator, validation_data=test_generator, steps_per_epoch=90, epochs=5)


model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
              loss='categorical_crossentropy', metrics=['accuracy'])

for epoch_num in [10, 15, 20, 25, 30, 35, 40, 45, 50]:
    model.fit_generator(
        train_generator, validation_data=test_generator, steps_per_epoch=90, epochs=5)
    model.save("./model/resnet_model_epoch{}.h5".format(epoch_num))
    print("model saved --> resnet_model_epoch{}.h5".format(epoch_num))
