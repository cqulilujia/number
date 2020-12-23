import os, sys
import numpy as np
import scipy
from scipy import ndimage
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.applications import InceptionV3
from keras.preprocessing import image
# from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
import random
from PIL import Image

INPUT_SIZE = 224


def DataSet():
    train_path = './data/train'
    category_list = os.listdir(train_path)
    for category in category_list:
        for


    return X_train, Y_train, X_test, Y_test


X_train, Y_train, X_test, Y_test = DataSet()
print('X_train shape : ', X_train.shape)
print('Y_train shape : ', Y_train.shape)
print('X_test shape : ', X_test.shape)
print('Y_test shape : ', Y_test.shape)


model = InceptionV3(
    weights=None,
    classes=10
)

model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# # train
model.fit(X_train, Y_train, epochs=1, batch_size=6)

# # evaluate
loss, acc = model.evaluate(X_test, Y_test, batch_size=32)
print(loss, acc)

# # save
model.save('save/my_inceptionv3_model.h5')
