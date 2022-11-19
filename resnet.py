import numpy as np
import pandas as pd
import cv2
from PIL import Image
import os
import pathlib
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.applications import InceptionResNetV2

from tensorflow.keras.models import Model
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout
import datetime
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


IMG_WIDTH = 244
IMG_HEIGHT = 244
IMG_DIM = (IMG_WIDTH, IMG_HEIGHT)
BATCH_SIZE = 32
IMG_DIR = pathlib.Path('data_class_generated')
TRAIN_DIR = 'data_class_generated/train'
VAL_DIR = 'data_class_generated/test'

from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Specify the values for all arguments to data_generator_with_aug.
data_generator_with_aug = ImageDataGenerator(preprocessing_function=preprocess_input,
                                             horizontal_flip=True,
                                             width_shift_range=0.2,
                                             height_shift_range=0.2,
                                             zoom_range=0.3,
                                             vertical_flip=True
                                             )

data_generator_no_aug = ImageDataGenerator(
    preprocessing_function=preprocess_input)

train_generator = data_generator_with_aug.flow_from_directory(
    directory=TRAIN_DIR,
    target_size=IMG_DIM,
    batch_size=BATCH_SIZE,
    class_mode='categorical')

validation_generator = data_generator_no_aug.flow_from_directory(
    directory=VAL_DIR,
    target_size=IMG_DIM, batch_size=BATCH_SIZE,
    class_mode='categorical')


nb_train = len(train_generator.filenames)
nb_val = len(validation_generator.filenames)

resnet = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(
    IMG_HEIGHT, IMG_WIDTH, 3), pooling='avg')

output = resnet.layers[-1].output
output = tf.keras.layers.Flatten()(output)
resnet = Model(resnet.input, output)

res_name = []
for layer in resnet.layers:
    res_name.append(layer.name)

# res_name[-447:]
num_classes = 3

model = Sequential()
model.add(resnet)
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(num_classes, activation='softmax'))

model.summary()
adam = tf.keras.optimizers.Adam(learning_rate=0.000001)
# Include the epoch in the file name (uses `str.format`)
checkpoint_path = 'my-model/checkpoint/cp-{epoch:04d}.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights every epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    save_freq='epoch')


early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                              restore_best_weights=False
                                              )


reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                 factor=0.3,
                                                 patience=5,
                                                 verbose=1,
                                                 min_delta=1e-3, min_lr=1e-7,
                                                 )

# set tensorboard
log_dir = "logs-resnet-class/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
callback_list = [reduce_lr, tensorboard_callback]


model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_generator, steps_per_epoch=int(nb_train // BATCH_SIZE),
          epochs=100, callbacks=[callback_list],
          validation_steps=int(nb_val / BATCH_SIZE),
          validation_data=validation_generator)
# model.fit(train_generator,
#           epochs=EPOCHS,
#           steps_per_epoch=train_num // BATCH_SIZE,
#           validation_data=valid_generator,
#           validation_steps=valid_num // BATCH_SIZE,
#           callbacks=callback_list,
#           verbose=0)


model.save('coffe-leaf-diseases-model-resnet.h5')
