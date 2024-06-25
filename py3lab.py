print("begin loading libraries")

import os
import tensorflow as tf
import numpy as np
from itertools import cycle
from PIL import Image

import time

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score

from numpy import argmax, array
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import callbacks
from keras.metrics import top_k_categorical_accuracy
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.utils import plot_model
from keras.metrics import top_k_categorical_accuracy

from keras.layers.convolutional import Conv2DTranspose, Conv2D

print("loaded all libraries")

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
   try:
	# Currently memory growth needs to be the same across GPUs
      for gpu in gpus:
         tf.config.experimental.set_memory_growth(gpu, True)
      #logical_gpus = tf.config.experimental.list_logical_devices('GPU')
      print(len(gpus), "Physical GPUs,")
      #, len(logical_gpus), "Logical GPUs")
   except RuntimeError as e:
      # Memory growth must be set before GPUs have been initialized
      print(e)

print("setup complete")

train_data_dir='/home/vlad/Desktop/train/'
validate_date_dir='/home/vlad/Desktop/allimages/'
test_data_dir='/home/vlad/Desktop/'

# Define constants
img_height = 250
img_width = 250

batch_size_train = 200
batch_size_test = 100

input_shape = (img_width, img_height, 1)

# Train Datagen
print("\nInitializing Training Data Generator: ")

print("bash script run successful")
