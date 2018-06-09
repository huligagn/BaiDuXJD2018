import tensorflow as tf
import os
import random
import tensorflow.contrib.eager as tfe
# Set Eager API
print("Setting Eager mode...")
tfe.enable_eager_execution()

TRAIN_PATH = '/home/huligang/workspace/BaiDuXJD2018/data/train/'
TEST_PATH = '/home/huligang/workspace/BaiDuXJD2018/data/test'

TARGET_NUM = 50
IMG_SIZE = 299

for catogery in os.listdir(TRAIN_PATH):
    img_dir = os.path.join(TRAIN_PATH, catogery)
    images = []
    for img in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img)
        images.append(img_path)

        image_raw_data = tf.gfile.FastGFile(img_path, 'rb').read()
        img_data = tf.image.decode_jpeg(image_raw_data)
        # padding
        height, width, _ = img_data.shape
        max_dim = int(max(height, width))
        img = tf.image.resize_image_with_crop_or_pad(
            img_data, max_dim, max_dim)
        
        img = tf.image.resize_images(img, (IMG_SIZE, IMG_SIZE))

        img = tf.cast(img, tf.uint8)
        try:
            img_raw_data = tf.image.encode_jpeg(img)
        except:
            print(img_path)

        assert img_path[-4:] == '.jpg'
        tf.gfile.FastGFile(
            img_path, 'wb').write(img_raw_data.numpy())

    print("aug class {}: {} --> {}".format(catogery, len(images), TARGET_NUM))
    for i in range(TARGET_NUM - len(images)):
        idx = random.randint(0, len(images)-1)
        image_raw_data = tf.gfile.FastGFile(images[idx], 'rb').read()
        img = tf.image.decode_jpeg(image_raw_data)
        
        # adjust contrast
        img = tf.image.random_contrast(img, 0.8, 1.2)
        img = tf.image.random_brightness(img,0.1)
        img = tf.image.resize_images(img, (IMG_SIZE,IMG_SIZE))

        img = tf.cast(img, tf.uint8)
        img_raw_data = tf.image.encode_jpeg(img)

        assert images[idx][-4:] == '.jpg'
        tf.gfile.FastGFile(images[idx][:-4] +"_{}_fake.jpg".format(i), 'wb').write(img_raw_data.numpy())

for catogery in os.listdir(TEST_PATH):
    img_dir = os.path.join(TEST_PATH, catogery)
    images = []
    for img in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img)
        images.append(img_path)

        image_raw_data = tf.gfile.FastGFile(img_path, 'rb').read()
        img_data = tf.image.decode_jpeg(image_raw_data)
        # padding
        height, width, _ = img_data.shape
        max_dim = int(max(height, width))
        img = tf.image.resize_image_with_crop_or_pad(
            img_data, max_dim, max_dim)

        img = tf.image.resize_images(img, (IMG_SIZE, IMG_SIZE))

        img = tf.cast(img, tf.uint8)
        img_raw_data = tf.image.encode_jpeg(img)

        assert img_path[-4:] == '.jpg'
        tf.gfile.FastGFile(
            img_path, 'wb').write(img_raw_data.numpy())
