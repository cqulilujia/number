import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# # restore
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model('save/my_resnet_model.h5')
INPUT_SIZE = 224
# # test
snow_file_path = './data/validation/snow'
half_snow_file_path = './data/validation/half_snow'
no_snow_file_path = './data/validation/no_snow'
snow_img_name_list = os.listdir(snow_file_path)
half_snow_img_name_list = os.listdir(half_snow_file_path)
no_snow_img_name_list = os.listdir(no_snow_file_path)
cnt_snow = cnt_half_snow = cnt_no_snow = 0
for img_name in snow_img_name_list:
    img_path = os.path.join(snow_file_path, img_name)
    img = image.load_img(img_path, target_size=(INPUT_SIZE, INPUT_SIZE))

    plt.imshow(img)
    img = image.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)  # 为batch添加第四维

    prediction = model.predict(img)
    if np.argmax(prediction) == 0:
        cnt_snow = cnt_snow + 1

for img_name in half_snow_img_name_list:
    img_path = os.path.join(half_snow_file_path, img_name)
    img = image.load_img(img_path, target_size=(INPUT_SIZE, INPUT_SIZE))

    plt.imshow(img)
    img = image.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)  # 为batch添加第四维

    prediction = model.predict(img)
    if np.argmax(prediction) == 1:
        cnt_half_snow = cnt_half_snow + 1

for img_name in no_snow_img_name_list:
    img_path = os.path.join(no_snow_file_path, img_name)
    img = image.load_img(img_path, target_size=(INPUT_SIZE, INPUT_SIZE))

    plt.imshow(img)
    img = image.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)  # 为batch添加第四维

    prediction = model.predict(img)
    if np.argmax(prediction) == 2:
        cnt_no_snow = cnt_no_snow + 1

print(cnt_snow / len(snow_img_name_list))
print(cnt_half_snow / len(half_snow_img_name_list))
print(cnt_no_snow / len(no_snow_img_name_list))
