print("begin loading libraries")


import os
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
print(len(gpus), "Physical GPUs, ")

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

from keras.preprocessing.image import ImageDataGenerator
from keras import callbacks
from keras.metrics import top_k_categorical_accuracy
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.utils import plot_model
from keras.metrics import top_k_categorical_accuracy

from tensorflow.keras.layers import Conv2DTranspose, Conv2D

print("loaded all libraries")

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

train_data_dir='/data/vlad.gaciu/beamProfilePublic/train/'
validate_data_dir='/data/vlad.gaciu/beamProfilePublic/validation/'
test_data_dir='/data/vlad.gaciu/beamProfilePublic/test/'

# Define constants
img_height = 400
img_width = 400

batch_size_train = 200
batch_size_test = 100
epochs=100

input_shape = (img_width, img_height, 1)

# Train Datagen
print("\nInitializing Training Data Generator: ")

train_datagen = ImageDataGenerator(
                                rescale=1. / 255,
                                #width_shift_range=2,
                                #height_shift_range=2,
                                #rotation_range=45,
                                #shear_range=1,
                                zoom_range=1,
                                featurewise_std_normalization=True)
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size_train,
    color_mode="grayscale",
    class_mode='categorical')
print("")

# Validation Datagen
print("Initializing Validation Data Generator:")
validation_datagen = ImageDataGenerator(
                                rescale=1. / 255,
                                #width_shift_range=2,
                                #height_shift_range=2,
                                #rotation_range=45,
                                #shear_range=1,
                                zoom_range=1,
                                featurewise_std_normalization=True)
valid_generator = validation_datagen.flow_from_directory(
    validate_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size_test,
    color_mode="grayscale",
    class_mode='categorical')
print("")

# Test Datagen
print("Initializing Test Data Generator:")
test_datagen = ImageDataGenerator(
                                rescale=1. / 255,
                                #width_shift_range=2,
                                #height_shift_range=2, 
                                #rotation_range=45,
                                #shear_range=1,
                                zoom_range=1,
                                featurewise_std_normalization=True)
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size_test,
    color_mode="grayscale",
    class_mode='categorical')
print("")

print("found all data")
print("\nbegin creating model")

#create model
model = Sequential()

print("model initiated")

#Encoder
model.add(Conv2D(27, (11, 11), input_shape=input_shape))
model.add(Activation('linear'))
model.add(MaxPooling2D(pool_size=(4, 4)))

model.add(Conv2D(9, (11, 11)))
model.add(Activation('linear'))
model.add(MaxPooling2D(pool_size=(4, 4)))
print("encoder initiated")

#Decoder

model.add(Conv2DTranspose(9, (13, 13)))
model.add(Activation('linear'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2DTranspose(27, (7, 7)))
model.add(Activation('linear'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

print("decoder initiated")

#Bottleneck

model.add(Conv2D(1, (4, 4)))
model.add(Activation('linear'))

print("bottleneck initiated")

#Output
model.add(Flatten())
model.add(Dense(9))
model.add(Activation('relu'))
model.add(Dense(3, activation='softmax'))
print("model complete")

output = model.summary()

#compile model
#num_classes = 3

#def top_3_categorical_accuracy(y_true, y_pred):
#    return top_k_categorical_accuracy(y_true, y_pred, k=10)

print("compile model")

model.compile(optimizer = tf.optimizers.SGD(learning_rate=0.00001),
              loss = 'SparseCategoricalCrossentropy',
              metrics = ['SparseCategoricalAccuracy'])

earlystopping = callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy', 
                                        mode='max', 
                                        patience=5, 
                                        restore_best_weights=True)
#fit model
print("Starting Training...")
history = model.fit(
        train_generator,
        steps_per_epoch=40,
        epochs=epochs,
        validation_data=test_generator,
        validation_steps=40,
        callbacks =[earlystopping],
        shuffle=True)
print("Done!\n\n")

model.evaluate(test_generator)

# Test model prediction accuracy against true data

path = '/data/vlad.gaciu/beamProfilPublic/validation/1/'
dirList = os.listdir(path)
plt.figure(figsize=(12,20))
count = 0
imageList = []
label = 'Low Confidence'

one = 0
two = 0
three = 0

for i in dirList[:60]:
    #print(i)
    image = Image.open(path + str(i))
    #plt.figure()
    #plt.imshow(image)
    #print(f"Original size : {image.size}")
        
    image = image.resize((img_width, img_height))
    #print(f"New size : {image.size}")

    x = np.asarray(image)   
    x = np.expand_dims(x, axis=0)
    #print(x/255.0)
    images = np.vstack([x])
    #print(images[0][50])
    classes = model.predict(
    images,
    batch_size=1,
    verbose='off',
    steps=None,
    callbacks=None,
    max_queue_size=10,
    workers=1,
    use_multiprocessing=True)

    classifier = "  Low Confidence"
    
    score = float(classes[0][0])
    #print(f"   This image is {100 * (score):.2f}% far-field.")
    if score > 0.66:
        classifier = f"{100 * (score):.2f}%  wedge"
        label = ['wedge']
        one += 1
   
    score = float(classes[0][1])
    #print(f"   This image is {100 * (score):.2f}% wedge.")
    if score > 0.66:
        classifier = f"{100 * (score):.2f}%  near-field"
        label = ['near-field']
        two += 1
    
    score = float(classes[0][2])
    #print(f"   This image is {100 * (score):.2f}% near-field.")
    if score > 0.66:
        classifier= f"{100 * (score):.2f}%  far-field"
        label = ['far-field']
        three += 1
    
    imageList += label
    
    
    if classifier == "  Low Confidence":
        classifier = f"{100 * (classes[0].max()):.2f}% Low Conf."
        
    if count < 60:
        ax=plt.subplot(10,6,count+1)
        plt.imshow(image, cmap='gray')
        plt.title(str(i[0:6] + "   " + classifier), fontsize = 7)
        count += 1
        #plt.axis('off')
print("Example Predicted Set:\n\n")

nameList = []
index = 0
while index < len(dirList[:]):
    if dirList[index][0:3] == 'B12':
        nameList += ['near-field']
    elif dirList[index][0:3] == 'B11':
        nameList += ['wedge']
    elif dirList[index][0:4] == 'ATW1':
        nameList += ['near-field']
    elif dirList[index][0:5] == 'FE2_C':
        nameList += ['wedge']
    elif dirList[index][0:5] == 'FE2_S':
        nameList += ['near-field']
    elif dirList[index][0:4] == 'BTW1':
        nameList += ['near-field']
    elif dirList[index][0:3] == 'A12':
        nameList += ['near-field']
    elif dirList[index][0:4] == 'ATW2':
        nameList += ['far-field']
    else:
        nameList += ['far-field']
    index += 1   

# Create confusion matrix to display results T/F
from sklearn.metrics import confusion_matrix

y_true = nameList
y_pred = imageList
confusion_matrix(y_true, y_pred)

from sklearn.metrics import ConfusionMatrixDisplay
cm = confusion_matrix(y_true, y_pred)

farTrue=cm[0][0]+cm[1][0]+cm[2][0]
nearTrue=cm[0][1]+cm[1][1]+cm[2][1]
wedgeTrue=cm[0][2]++cm[1][2]+cm[2][2]

correctValue = cm[0][0]+cm[1][1]+cm[2][2]

print(f"High Confidence Images: {100*(one+two+three)/(index):.2f} % of total (greater than 66% confidence value).")
print("\n")

print(f"{(one)}/{(wedgeTrue)} wedge predictions: 0-")
print(f"{(two)}/{(nearTrue)} near-field predictions: 1-")
print(f"{(three)}/{(farTrue)} far-field predictions: 2-")


print("\n")
print(f"{correctValue} values correct out of {index}: {100 * (correctValue/index):.2f}% model accuracy")
#cm_display = ConfusionMatrixDisplay(cm).plot()


print("bash script run successful")
