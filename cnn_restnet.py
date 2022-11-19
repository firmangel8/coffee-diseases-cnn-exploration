from gc import callbacks
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from PIL import Image
import tensorflow as tf


import os

from matplotlib import pyplot
from matplotlib.image import imread

from distutils.dir_util import copy_tree
from shutil import copy

from tensorflow.keras import layers, models

from matplotlib import pyplot as plt

import keras as keras

from keras.preprocessing import image

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import datetime
import tensorflow as tf
import tensorflow.keras.metrics as tfm
gpu = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True) #limits gpu memory


class AUC(tfm.AUC):
    def __init__(self, from_logits=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._from_logits = from_logits

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self._from_logits:
            super(AUC, self).update_state(
                y_true, tf.nn.sigmoid(y_pred), sample_weight)
        else:
            super(AUC, self).update_state(y_true, y_pred, sample_weight)


class BinaryAccuracy(tfm.BinaryAccuracy):
    def __init__(self, from_logits=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._from_logits = from_logits

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self._from_logits:
            super(BinaryAccuracy, self).update_state(
                y_true, tf.nn.sigmoid(y_pred), sample_weight)
        else:
            super(BinaryAccuracy, self).update_state(
                y_true, y_pred, sample_weight)


class TruePositives(tfm.TruePositives):
    def __init__(self, from_logits=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._from_logits = from_logits

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self._from_logits:
            super(TruePositives, self).update_state(
                y_true, tf.nn.sigmoid(y_pred), sample_weight)
        else:
            super(TruePositives, self).update_state(
                y_true, y_pred, sample_weight)


class FalsePositives(tfm.FalsePositives):
    def __init__(self, from_logits=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._from_logits = from_logits

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self._from_logits:
            super(FalsePositives, self).update_state(
                y_true, tf.nn.sigmoid(y_pred), sample_weight)
        else:
            super(FalsePositives, self).update_state(
                y_true, y_pred, sample_weight)


class TrueNegatives(tfm.TrueNegatives):
    def __init__(self, from_logits=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._from_logits = from_logits

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self._from_logits:
            super(TrueNegatives, self).update_state(
                y_true, tf.nn.sigmoid(y_pred), sample_weight)
        else:
            super(TrueNegatives, self).update_state(
                y_true, y_pred, sample_weight)


class FalseNegatives(tfm.FalseNegatives):
    def __init__(self, from_logits=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._from_logits = from_logits

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self._from_logits:
            super(FalseNegatives, self).update_state(
                y_true, tf.nn.sigmoid(y_pred), sample_weight)
        else:
            super(FalseNegatives, self).update_state(
                y_true, y_pred, sample_weight)


class Precision(tfm.Precision):
    def __init__(self, from_logits=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._from_logits = from_logits

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self._from_logits:
            super(Precision, self).update_state(
                y_true, tf.nn.sigmoid(y_pred), sample_weight)
        else:
            super(Precision, self).update_state(y_true, y_pred, sample_weight)


class Recall(tfm.Recall):
    def __init__(self, from_logits=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._from_logits = from_logits

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self._from_logits:
            super(Recall, self).update_state(
                y_true, tf.nn.sigmoid(y_pred), sample_weight)
        else:
            super(Recall, self).update_state(y_true, y_pred, sample_weight)


traindatasetm = tf.keras.preprocessing.image_dataset_from_directory(
    'data_segmented/train/imagesm',
    labels='inferred',
    label_mode="binary",
    class_names=None,
    color_mode="rgb",
    batch_size=16,
    image_size=(512, 512),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False
)


testdatasetm = tf.keras.preprocessing.image_dataset_from_directory(
    'data_segmented/test/imagesm',
    labels='inferred',
    label_mode="binary",
    class_names=None,
    color_mode="rgb",
    batch_size=16,
    image_size=(512, 512),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False
)

base_model = ResNet50(weights='imagenet', include_top=False,
                      input_shape=(512, 512, 3))
base_model.trainable = False

inputs = keras.Input(shape=(512, 512, 3))
x = keras.layers.RandomFlip(mode="horizontal_and_vertical")(inputs)
x = keras.layers.RandomTranslation(
    height_factor=0.1, width_factor=0.3, fill_mode="reflect", interpolation="bilinear")(x)
x = tf.keras.layers.RandomContrast(factor=0.2)(x)
x = preprocess_input(x)
x = base_model(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = keras.layers.Dense(1)(x)
modelm = keras.Model(inputs, outputs)
modelm.summary()

log_dir = "logs-resnet-train/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callbackx = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
modelm.compile(optimizer='adam',
               loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
               metrics=['accuracy', Precision(from_logits=True), Recall(from_logits=True)])
historym = modelm.fit(x=traindatasetm, epochs=80, validation_data=testdatasetm, callbacks=[tensorboard_callbackx])

base_model.trainable = True


print("Number of layers in the base model: ", len(base_model.layers))

# Fine-tune from this layer onwards
fine_tune_at = 80

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False


modelm.summary()

modelm.compile(
    optimizer=tf.keras.optimizers.Adam(1e-6),  # Low learning rate
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy', Precision(from_logits=True), Recall(from_logits=True)]
)
log_dir = "logs-resnet-support-train/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
historym2 = modelm.fit(x=traindatasetm, epochs=20,
                       validation_data=testdatasetm, callbacks=[tensorboard_callback])
modelm.save('binminer.h5')


acc_list = historym.history['accuracy'] + historym2.history['accuracy']
acc = np.array(acc_list)
# plt.plot(acc, label='accuracy')
val_acc = historym.history['val_accuracy'] + historym2.history['val_accuracy']
# plt.plot(val_acc, label = 'val_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.ylim([0.6, 1])
# plt.legend(loc='lower right')
# plt.show()

acc_list = historym.history['precision'] + historym2.history['precision_1']
acc = np.array(acc_list)
# plt.plot(acc, label='precision')
val_acc = historym.history['val_precision'] + \
    historym2.history['val_precision_1']
# plt.plot(val_acc, label = 'val_precision')
# plt.xlabel('Epoch')
# plt.ylabel('Precision')
# plt.ylim([0.6, 1])
# plt.legend(loc='lower right')
# plt.show()

acc_list = historym.history['recall'] + historym2.history['recall_1']
acc = np.array(acc_list)
# plt.plot(acc, label='recall')
val_acc = historym.history['val_recall'] + historym2.history['val_recall_1']
# plt.plot(val_acc, label = 'val_recall')
# plt.xlabel('Epoch')
# plt.ylabel('Recall')
# plt.ylim([0.6, 1])
# plt.legend(loc='lower right')
# plt.show()
