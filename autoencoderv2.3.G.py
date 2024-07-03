{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "97684d4c",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2023-07-25 14:33:01.028066: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA\n",
      "To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.\n",
      "2023-07-25 14:33:01.171937: E tensorflow/stream_executor/cuda/cuda_blas.cc:2981] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered\n",
      "2023-07-25 14:33:01.654975: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libnvinfer.so.7'; dlerror: libnvinfer.so.7: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /home/vlad/root/lib:/home/vlad/anaconda3/lib/:/home/vlad/anaconda3/lib/python3.9/site-packages/nvidia/cudnn/lib\n",
      "2023-07-25 14:33:01.655040: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libnvinfer_plugin.so.7'; dlerror: libnvinfer_plugin.so.7: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /home/vlad/root/lib:/home/vlad/anaconda3/lib/:/home/vlad/anaconda3/lib/python3.9/site-packages/nvidia/cudnn/lib\n",
      "2023-07-25 14:33:01.655046: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Cannot dlopen some TensorRT libraries. If you would like to use Nvidia GPU with TensorRT, please make sure the missing libraries mentioned above are installed properly.\n",
      "/home/vlad/anaconda3/lib/python3.9/site-packages/scipy/__init__.py:146: UserWarning: A NumPy version >=1.16.5 and <1.23.0 is required for this version of SciPy (detected version 1.23.5\n",
      "  warnings.warn(f\"A NumPy version >={np_minversion} and <{np_maxversion}\"\n"
     ]
    }
   ],
   "source": [
    "import os\n",
    "import tensorflow as tf\n",
    "import numpy as np\n",
    "from itertools import cycle\n",
    "from PIL import Image\n",
    "\n",
    "import time\n",
    "\n",
    "from sklearn import svm, datasets\n",
    "from sklearn.metrics import roc_curve, auc\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import label_binarize\n",
    "from sklearn.multiclass import OneVsRestClassifier\n",
    "from sklearn.metrics import roc_auc_score\n",
    "\n",
    "from numpy import argmax, array\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "from sklearn.preprocessing import OneHotEncoder\n",
    "from keras.models import Sequential\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib.image as mpimg\n",
    "\n",
    "from tensorflow.keras.preprocessing.image import ImageDataGenerator\n",
    "from keras import callbacks\n",
    "from keras.metrics import top_k_categorical_accuracy\n",
    "from keras.models import Sequential\n",
    "from keras.layers import Conv2D, MaxPooling2D\n",
    "from keras.layers import Activation, Dropout, Flatten, Dense\n",
    "from keras import backend as K\n",
    "from keras.utils import plot_model\n",
    "from keras.metrics import top_k_categorical_accuracy\n",
    "\n",
    "from keras.layers.convolutional import Conv2DTranspose , Conv2D"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "785268ba",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1 Physical GPUs, 1 Logical GPUs\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2023-07-25 14:33:02.840938: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:980] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero\n",
      "2023-07-25 14:33:02.868493: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:980] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero\n",
      "2023-07-25 14:33:02.870549: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:980] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero\n",
      "2023-07-25 14:33:02.873202: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA\n",
      "To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.\n",
      "2023-07-25 14:33:02.873923: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:980] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero\n",
      "2023-07-25 14:33:02.891281: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:980] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero\n",
      "2023-07-25 14:33:02.898129: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:980] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero\n",
      "2023-07-25 14:33:03.646128: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:980] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero\n",
      "2023-07-25 14:33:03.647193: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:980] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero\n",
      "2023-07-25 14:33:03.648144: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:980] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero\n",
      "2023-07-25 14:33:03.649114: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1616] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 5038 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 3070, pci bus id: 0000:09:00.0, compute capability: 8.6\n"
     ]
    }
   ],
   "source": [
    "gpus = tf.config.experimental.list_physical_devices('GPU')\n",
    "if gpus:\n",
    "    try:\n",
    "        # Currently, memory growth needs to be the same across GPUs\n",
    "        for gpu in gpus:\n",
    "            tf.config.experimental.set_memory_growth(gpu, True)\n",
    "        logical_gpus = tf.config.experimental.list_logical_devices('GPU')\n",
    "        print(len(gpus), \"Physical GPUs,\", len(logical_gpus), \"Logical GPUs\")\n",
    "    except RuntimeError as e:\n",
    "        # Memory growth must be set before GPUs have been initialized\n",
    "        print(e)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "09859948",
   "metadata": {},
   "outputs": [],
   "source": [
    "train_data_dir='/home/vlad/Desktop/train_2/'\n",
    "validate_data_dir='/home/vlad/Desktop/allimages_2/'\n",
    "test_data_dir='/home/vlad/Desktop/test_2/'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "301764e4",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Define Constants\n",
    "img_height = 250\n",
    "img_width = 250\n",
    "\n",
    "batch_size_train=200\n",
    "batch_size_test=100\n",
    "epochs=100"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "977da7ca",
   "metadata": {},
   "outputs": [],
   "source": [
    "input_shape = (img_width, img_height, 1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "6976f0a3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Initializing Training Data Generator:\n",
      "Found 8760 images belonging to 1 classes.\n",
      "\n",
      "Initializing Validation Data Generator:\n",
      "Found 1459 images belonging to 1 classes.\n",
      "\n",
      "Initializing Test Data Generator:\n",
      "Found 4380 images belonging to 1 classes.\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/home/vlad/anaconda3/lib/python3.9/site-packages/keras/preprocessing/image.py:1462: UserWarning: This ImageDataGenerator specifies `featurewise_std_normalization`, which overrides setting of `featurewise_center`.\n",
      "  warnings.warn(\n"
     ]
    }
   ],
   "source": [
    "\n",
    "\n",
    "# Train Datagen\n",
    "print(\"\\nInitializing Training Data Generator:\")\n",
    "train_datagen = ImageDataGenerator(\n",
    "                                rescale=1. / 255,\n",
    "                                #width_shift_range=2,\n",
    "                                #height_shift_range=2,\n",
    "                                #rotation_range=45,\n",
    "                                #shear_range=1,\n",
    "                                zoom_range=1,\n",
    "                                featurewise_std_normalization=True)\n",
    "train_generator = train_datagen.flow_from_directory(\n",
    "    train_data_dir,\n",
    "    target_size=(img_width, img_height),\n",
    "    batch_size=batch_size_train,\n",
    "    color_mode=\"grayscale\",\n",
    "    class_mode='categorical')\n",
    "print(\"\")\n",
    "\n",
    "# Validation Datagen\n",
    "print(\"Initializing Validation Data Generator:\")\n",
    "validation_datagen = ImageDataGenerator(\n",
    "                                rescale=1. / 255,\n",
    "                                #width_shift_range=2,\n",
    "                                #height_shift_range=2,\n",
    "                                #rotation_range=45,\n",
    "                                #shear_range=1,\n",
    "                                zoom_range=1,\n",
    "                                featurewise_std_normalization=True)\n",
    "valid_generator = validation_datagen.flow_from_directory(\n",
    "    validate_data_dir,\n",
    "    target_size=(img_width, img_height),\n",
    "    batch_size=batch_size_test,\n",
    "    color_mode=\"grayscale\",\n",
    "    class_mode='categorical')\n",
    "print(\"\")\n",
    "\n",
    "# Test Datagen\n",
    "print(\"Initializing Test Data Generator:\")\n",
    "test_datagen = ImageDataGenerator(\n",
    "                                rescale=1. / 255,\n",
    "                                #width_shift_range=2,\n",
    "                                #height_shift_range=2, \n",
    "                                #rotation_range=45,\n",
    "                                #shear_range=1,\n",
    "                                zoom_range=1,\n",
    "                                featurewise_std_normalization=True)\n",
    "test_generator = test_datagen.flow_from_directory(\n",
    "    test_data_dir,\n",
    "    target_size=(img_width, img_height),\n",
    "    batch_size=batch_size_test,\n",
    "    color_mode=\"grayscale\",\n",
    "    class_mode='categorical')\n",
    "print(\"\")\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "5a35df19",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"sequential\"\n",
      "_________________________________________________________________\n",
      " Layer (type)                Output Shape              Param #   \n",
      "=================================================================\n",
      " conv2d (Conv2D)             (None, 240, 240, 27)      3294      \n",
      "                                                                 \n",
      " activation (Activation)     (None, 240, 240, 27)      0         \n",
      "                                                                 \n",
      " max_pooling2d (MaxPooling2D  (None, 60, 60, 27)       0         \n",
      " )                                                               \n",
      "                                                                 \n",
      " conv2d_1 (Conv2D)           (None, 50, 50, 9)         29412     \n",
      "                                                                 \n",
      " activation_1 (Activation)   (None, 50, 50, 9)         0         \n",
      "                                                                 \n",
      " max_pooling2d_1 (MaxPooling  (None, 12, 12, 9)        0         \n",
      " 2D)                                                             \n",
      "                                                                 \n",
      " conv2d_transpose (Conv2DTra  (None, 24, 24, 9)        13698     \n",
      " nspose)                                                         \n",
      "                                                                 \n",
      " activation_2 (Activation)   (None, 24, 24, 9)         0         \n",
      "                                                                 \n",
      " conv2d_transpose_1 (Conv2DT  (None, 30, 30, 27)       11934     \n",
      " ranspose)                                                       \n",
      "                                                                 \n",
      " activation_3 (Activation)   (None, 30, 30, 27)        0         \n",
      "                                                                 \n",
      " conv2d_2 (Conv2D)           (None, 27, 27, 1)         433       \n",
      "                                                                 \n",
      " activation_4 (Activation)   (None, 27, 27, 1)         0         \n",
      "                                                                 \n",
      " flatten (Flatten)           (None, 729)               0         \n",
      "                                                                 \n",
      " dense (Dense)               (None, 9)                 6570      \n",
      "                                                                 \n",
      " activation_5 (Activation)   (None, 9)                 0         \n",
      "                                                                 \n",
      " dense_1 (Dense)             (None, 3)                 30        \n",
      "                                                                 \n",
      "=================================================================\n",
      "Total params: 65,371\n",
      "Trainable params: 65,371\n",
      "Non-trainable params: 0\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "#create model\n",
    "model = Sequential()\n",
    "\n",
    "#Encoder\n",
    "model.add(Conv2D(27, (11, 11), input_shape=input_shape))\n",
    "model.add(Activation('linear'))\n",
    "model.add(MaxPooling2D(pool_size=(4, 4)))\n",
    "\n",
    "model.add(Conv2D(9, (11, 11)))\n",
    "model.add(Activation('linear'))\n",
    "model.add(MaxPooling2D(pool_size=(4, 4)))\n",
    "\n",
    "#Decoder\n",
    "\n",
    "model.add(Conv2DTranspose(9, (13, 13)))\n",
    "model.add(Activation('linear'))\n",
    "#model.add(MaxPooling2D(pool_size=(2, 2)))\n",
    "\n",
    "model.add(Conv2DTranspose(27, (7, 7)))\n",
    "model.add(Activation('linear'))\n",
    "#model.add(MaxPooling2D(pool_size=(2, 2)))\n",
    "\n",
    "#Bottleneck\n",
    "\n",
    "model.add(Conv2D(1, (4, 4)))\n",
    "model.add(Activation('linear'))\n",
    "\n",
    "#Output\n",
    "model.add(Flatten())\n",
    "model.add(Dense(9))\n",
    "model.add(Activation('relu'))\n",
    "model.add(Dense(3, activation='softmax'))\n",
    "\n",
    "output = model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "d2bb4522",
   "metadata": {},
   "outputs": [],
   "source": [
    "#compile model\n",
    "#num_classes = 3\n",
    "\n",
    "#def top_3_categorical_accuracy(y_true, y_pred):\n",
    "#    return top_k_categorical_accuracy(y_true, y_pred, k=10)\n",
    "\n",
    "model.compile(optimizer = tf.optimizers.SGD(learning_rate=0.00001),\n",
    "              loss = 'SparseCategoricalCrossentropy',\n",
    "              metrics = ['SparseCategoricalAccuracy'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "207217da",
   "metadata": {},
   "outputs": [],
   "source": [
    "earlystopping = callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy', \n",
    "                                        mode='max', \n",
    "                                        patience=5, \n",
    "                                        restore_best_weights=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "122dc713",
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Starting Training...\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/home/vlad/anaconda3/lib/python3.9/site-packages/keras/preprocessing/image.py:1863: UserWarning: This ImageDataGenerator specifies `featurewise_center`, but it hasn't been fit on any training data. Fit it first by calling `.fit(numpy_data)`.\n",
      "  warnings.warn(\n",
      "/home/vlad/anaconda3/lib/python3.9/site-packages/keras/preprocessing/image.py:1873: UserWarning: This ImageDataGenerator specifies `featurewise_std_normalization`, but it hasn't been fit on any training data. Fit it first by calling `.fit(numpy_data)`.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/100\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2023-07-25 14:33:07.115289: I tensorflow/stream_executor/cuda/cuda_dnn.cc:384] Loaded cuDNN version 8600\n",
      "2023-07-25 14:33:08.311577: I tensorflow/stream_executor/cuda/cuda_blas.cc:1614] TensorFloat-32 will be used for the matrix multiplication. This will only be logged once.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "40/40 [==============================] - 45s 981ms/step - loss: 1.1119 - sparse_categorical_accuracy: 0.0601 - val_loss: 1.1113 - val_sparse_categorical_accuracy: 0.0645\n",
      "Epoch 2/100\n",
      "40/40 [==============================] - 41s 1s/step - loss: 1.1110 - sparse_categorical_accuracy: 0.0722 - val_loss: 1.1104 - val_sparse_categorical_accuracy: 0.0730\n",
      "Epoch 3/100\n",
      "40/40 [==============================] - 40s 1s/step - loss: 1.1101 - sparse_categorical_accuracy: 0.0838 - val_loss: 1.1094 - val_sparse_categorical_accuracy: 0.0838\n",
      "Epoch 4/100\n",
      "40/40 [==============================] - 40s 1s/step - loss: 1.1088 - sparse_categorical_accuracy: 0.0948 - val_loss: 1.1081 - val_sparse_categorical_accuracy: 0.0970\n",
      "Epoch 5/100\n",
      "40/40 [==============================] - 40s 993ms/step - loss: 1.1078 - sparse_categorical_accuracy: 0.1129 - val_loss: 1.1070 - val_sparse_categorical_accuracy: 0.1268\n",
      "Epoch 6/100\n",
      "40/40 [==============================] - 40s 1s/step - loss: 1.1071 - sparse_categorical_accuracy: 0.1276 - val_loss: 1.1062 - val_sparse_categorical_accuracy: 0.1230\n",
      "Epoch 7/100\n",
      "40/40 [==============================] - 40s 1s/step - loss: 1.1062 - sparse_categorical_accuracy: 0.1372 - val_loss: 1.1057 - val_sparse_categorical_accuracy: 0.1353\n",
      "Epoch 8/100\n",
      "40/40 [==============================] - 42s 1s/step - loss: 1.1052 - sparse_categorical_accuracy: 0.1597 - val_loss: 1.1048 - val_sparse_categorical_accuracy: 0.1525\n",
      "Epoch 9/100\n",
      "40/40 [==============================] - 41s 1s/step - loss: 1.1045 - sparse_categorical_accuracy: 0.1706 - val_loss: 1.1039 - val_sparse_categorical_accuracy: 0.1755\n",
      "Epoch 10/100\n",
      "40/40 [==============================] - 40s 1s/step - loss: 1.1036 - sparse_categorical_accuracy: 0.1856 - val_loss: 1.1031 - val_sparse_categorical_accuracy: 0.1935\n",
      "Epoch 11/100\n",
      "40/40 [==============================] - 41s 1s/step - loss: 1.1030 - sparse_categorical_accuracy: 0.2090 - val_loss: 1.1022 - val_sparse_categorical_accuracy: 0.2048\n",
      "Epoch 12/100\n",
      "40/40 [==============================] - 40s 1s/step - loss: 1.1020 - sparse_categorical_accuracy: 0.2222 - val_loss: 1.1015 - val_sparse_categorical_accuracy: 0.2282\n",
      "Epoch 13/100\n",
      "40/40 [==============================] - 40s 1s/step - loss: 1.1015 - sparse_categorical_accuracy: 0.2489 - val_loss: 1.1012 - val_sparse_categorical_accuracy: 0.2425\n",
      "Epoch 14/100\n",
      "40/40 [==============================] - 41s 1s/step - loss: 1.1007 - sparse_categorical_accuracy: 0.2606 - val_loss: 1.1003 - val_sparse_categorical_accuracy: 0.2690\n",
      "Epoch 15/100\n",
      "40/40 [==============================] - 41s 1s/step - loss: 1.1005 - sparse_categorical_accuracy: 0.2842 - val_loss: 1.0995 - val_sparse_categorical_accuracy: 0.2920\n",
      "Epoch 16/100\n",
      "40/40 [==============================] - 40s 1s/step - loss: 1.0995 - sparse_categorical_accuracy: 0.2976 - val_loss: 1.0991 - val_sparse_categorical_accuracy: 0.3110\n",
      "Epoch 17/100\n",
      "40/40 [==============================] - 41s 1s/step - loss: 1.0989 - sparse_categorical_accuracy: 0.3227 - val_loss: 1.0985 - val_sparse_categorical_accuracy: 0.3223\n",
      "Epoch 18/100\n",
      "40/40 [==============================] - 40s 1s/step - loss: 1.0983 - sparse_categorical_accuracy: 0.3538 - val_loss: 1.0978 - val_sparse_categorical_accuracy: 0.3505\n",
      "Epoch 19/100\n",
      "40/40 [==============================] - 40s 1s/step - loss: 1.0978 - sparse_categorical_accuracy: 0.3584 - val_loss: 1.0971 - val_sparse_categorical_accuracy: 0.3770\n",
      "Epoch 20/100\n",
      "40/40 [==============================] - 41s 1s/step - loss: 1.0971 - sparse_categorical_accuracy: 0.3893 - val_loss: 1.0966 - val_sparse_categorical_accuracy: 0.3918\n",
      "Epoch 21/100\n",
      "40/40 [==============================] - 41s 1s/step - loss: 1.0965 - sparse_categorical_accuracy: 0.4085 - val_loss: 1.0959 - val_sparse_categorical_accuracy: 0.4280\n",
      "Epoch 22/100\n",
      "40/40 [==============================] - 41s 1s/step - loss: 1.0958 - sparse_categorical_accuracy: 0.4226 - val_loss: 1.0955 - val_sparse_categorical_accuracy: 0.4305\n",
      "Epoch 23/100\n",
      "40/40 [==============================] - 40s 1s/step - loss: 1.0953 - sparse_categorical_accuracy: 0.4457 - val_loss: 1.0952 - val_sparse_categorical_accuracy: 0.4442\n",
      "Epoch 24/100\n",
      "40/40 [==============================] - 41s 1s/step - loss: 1.0947 - sparse_categorical_accuracy: 0.4580 - val_loss: 1.0944 - val_sparse_categorical_accuracy: 0.4795\n",
      "Epoch 25/100\n",
      "40/40 [==============================] - 40s 1s/step - loss: 1.0943 - sparse_categorical_accuracy: 0.4799 - val_loss: 1.0939 - val_sparse_categorical_accuracy: 0.4915\n",
      "Epoch 26/100\n",
      "40/40 [==============================] - 41s 1s/step - loss: 1.0938 - sparse_categorical_accuracy: 0.4956 - val_loss: 1.0932 - val_sparse_categorical_accuracy: 0.5065\n",
      "Epoch 27/100\n",
      "40/40 [==============================] - 41s 1s/step - loss: 1.0930 - sparse_categorical_accuracy: 0.5090 - val_loss: 1.0926 - val_sparse_categorical_accuracy: 0.5295\n",
      "Epoch 28/100\n",
      "40/40 [==============================] - 41s 1s/step - loss: 1.0924 - sparse_categorical_accuracy: 0.5392 - val_loss: 1.0921 - val_sparse_categorical_accuracy: 0.5282\n",
      "Epoch 29/100\n",
      "40/40 [==============================] - 40s 1s/step - loss: 1.0919 - sparse_categorical_accuracy: 0.5499 - val_loss: 1.0917 - val_sparse_categorical_accuracy: 0.5663\n",
      "Epoch 30/100\n",
      "40/40 [==============================] - 41s 1s/step - loss: 1.0913 - sparse_categorical_accuracy: 0.5693 - val_loss: 1.0914 - val_sparse_categorical_accuracy: 0.5642\n",
      "Epoch 31/100\n",
      "40/40 [==============================] - 40s 1s/step - loss: 1.0908 - sparse_categorical_accuracy: 0.5810 - val_loss: 1.0909 - val_sparse_categorical_accuracy: 0.5723\n",
      "Epoch 32/100\n",
      "40/40 [==============================] - 40s 1s/step - loss: 1.0905 - sparse_categorical_accuracy: 0.5845 - val_loss: 1.0904 - val_sparse_categorical_accuracy: 0.5842\n",
      "Epoch 33/100\n",
      "40/40 [==============================] - 41s 1s/step - loss: 1.0899 - sparse_categorical_accuracy: 0.5991 - val_loss: 1.0895 - val_sparse_categorical_accuracy: 0.6133\n",
      "Epoch 34/100\n",
      "40/40 [==============================] - 40s 1s/step - loss: 1.0895 - sparse_categorical_accuracy: 0.6201 - val_loss: 1.0889 - val_sparse_categorical_accuracy: 0.6385\n",
      "Epoch 35/100\n",
      "40/40 [==============================] - 40s 1s/step - loss: 1.0889 - sparse_categorical_accuracy: 0.6231 - val_loss: 1.0884 - val_sparse_categorical_accuracy: 0.6370\n",
      "Epoch 36/100\n",
      "40/40 [==============================] - 42s 1s/step - loss: 1.0882 - sparse_categorical_accuracy: 0.6460 - val_loss: 1.0882 - val_sparse_categorical_accuracy: 0.6497\n",
      "Epoch 37/100\n",
      "40/40 [==============================] - 41s 1s/step - loss: 1.0878 - sparse_categorical_accuracy: 0.6456 - val_loss: 1.0879 - val_sparse_categorical_accuracy: 0.6515\n",
      "Epoch 38/100\n",
      "40/40 [==============================] - 41s 1s/step - loss: 1.0873 - sparse_categorical_accuracy: 0.6634 - val_loss: 1.0875 - val_sparse_categorical_accuracy: 0.6672\n",
      "Epoch 39/100\n",
      "40/40 [==============================] - 40s 1s/step - loss: 1.0865 - sparse_categorical_accuracy: 0.6813 - val_loss: 1.0867 - val_sparse_categorical_accuracy: 0.6708\n",
      "Epoch 40/100\n",
      "40/40 [==============================] - 40s 1s/step - loss: 1.0863 - sparse_categorical_accuracy: 0.6893 - val_loss: 1.0860 - val_sparse_categorical_accuracy: 0.6935\n",
      "Epoch 41/100\n",
      "40/40 [==============================] - 41s 1s/step - loss: 1.0858 - sparse_categorical_accuracy: 0.6945 - val_loss: 1.0854 - val_sparse_categorical_accuracy: 0.7042\n",
      "Epoch 42/100\n",
      "40/40 [==============================] - 41s 1s/step - loss: 1.0851 - sparse_categorical_accuracy: 0.7138 - val_loss: 1.0853 - val_sparse_categorical_accuracy: 0.7125\n",
      "Epoch 43/100\n",
      "40/40 [==============================] - 41s 1s/step - loss: 1.0847 - sparse_categorical_accuracy: 0.7193 - val_loss: 1.0844 - val_sparse_categorical_accuracy: 0.7203\n",
      "Epoch 44/100\n",
      "40/40 [==============================] - 40s 1s/step - loss: 1.0845 - sparse_categorical_accuracy: 0.7182 - val_loss: 1.0843 - val_sparse_categorical_accuracy: 0.7315\n",
      "Epoch 45/100\n",
      "40/40 [==============================] - 40s 1s/step - loss: 1.0837 - sparse_categorical_accuracy: 0.7406 - val_loss: 1.0835 - val_sparse_categorical_accuracy: 0.7423\n",
      "Epoch 46/100\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "40/40 [==============================] - 42s 1s/step - loss: 1.0832 - sparse_categorical_accuracy: 0.7485 - val_loss: 1.0827 - val_sparse_categorical_accuracy: 0.7527\n",
      "Epoch 47/100\n",
      "40/40 [==============================] - 40s 998ms/step - loss: 1.0827 - sparse_categorical_accuracy: 0.7583 - val_loss: 1.0822 - val_sparse_categorical_accuracy: 0.7685\n",
      "Epoch 48/100\n",
      "40/40 [==============================] - 40s 1s/step - loss: 1.0823 - sparse_categorical_accuracy: 0.7599 - val_loss: 1.0822 - val_sparse_categorical_accuracy: 0.7697\n",
      "Epoch 49/100\n",
      "40/40 [==============================] - 41s 1s/step - loss: 1.0817 - sparse_categorical_accuracy: 0.7730 - val_loss: 1.0817 - val_sparse_categorical_accuracy: 0.7697\n",
      "Epoch 50/100\n",
      "40/40 [==============================] - 42s 1s/step - loss: 1.0815 - sparse_categorical_accuracy: 0.7709 - val_loss: 1.0810 - val_sparse_categorical_accuracy: 0.7872\n",
      "Epoch 51/100\n",
      "40/40 [==============================] - 40s 1s/step - loss: 1.0805 - sparse_categorical_accuracy: 0.7876 - val_loss: 1.0807 - val_sparse_categorical_accuracy: 0.7922\n",
      "Epoch 52/100\n",
      "40/40 [==============================] - 41s 1s/step - loss: 1.0804 - sparse_categorical_accuracy: 0.7851 - val_loss: 1.0801 - val_sparse_categorical_accuracy: 0.7990\n",
      "Epoch 53/100\n",
      "40/40 [==============================] - 41s 1s/step - loss: 1.0798 - sparse_categorical_accuracy: 0.8031 - val_loss: 1.0797 - val_sparse_categorical_accuracy: 0.8052\n",
      "Epoch 54/100\n",
      "40/40 [==============================] - 41s 1s/step - loss: 1.0794 - sparse_categorical_accuracy: 0.8025 - val_loss: 1.0789 - val_sparse_categorical_accuracy: 0.8145\n",
      "Epoch 55/100\n",
      "40/40 [==============================] - 40s 1s/step - loss: 1.0787 - sparse_categorical_accuracy: 0.8207 - val_loss: 1.0786 - val_sparse_categorical_accuracy: 0.8235\n",
      "Epoch 56/100\n",
      "40/40 [==============================] - 40s 1s/step - loss: 1.0780 - sparse_categorical_accuracy: 0.8265 - val_loss: 1.0780 - val_sparse_categorical_accuracy: 0.8280\n",
      "Epoch 57/100\n",
      "40/40 [==============================] - 40s 1s/step - loss: 1.0776 - sparse_categorical_accuracy: 0.8330 - val_loss: 1.0778 - val_sparse_categorical_accuracy: 0.8298\n",
      "Epoch 58/100\n",
      "40/40 [==============================] - 42s 1s/step - loss: 1.0774 - sparse_categorical_accuracy: 0.8338 - val_loss: 1.0770 - val_sparse_categorical_accuracy: 0.8367\n",
      "Epoch 59/100\n",
      "40/40 [==============================] - 41s 1s/step - loss: 1.0766 - sparse_categorical_accuracy: 0.8353 - val_loss: 1.0766 - val_sparse_categorical_accuracy: 0.8372\n",
      "Epoch 60/100\n",
      "40/40 [==============================] - 41s 1s/step - loss: 1.0761 - sparse_categorical_accuracy: 0.8389 - val_loss: 1.0761 - val_sparse_categorical_accuracy: 0.8600\n",
      "Epoch 61/100\n",
      "40/40 [==============================] - 40s 1s/step - loss: 1.0756 - sparse_categorical_accuracy: 0.8513 - val_loss: 1.0754 - val_sparse_categorical_accuracy: 0.8525\n",
      "Epoch 62/100\n",
      "40/40 [==============================] - 41s 1s/step - loss: 1.0751 - sparse_categorical_accuracy: 0.8544 - val_loss: 1.0750 - val_sparse_categorical_accuracy: 0.8572\n",
      "Epoch 63/100\n",
      "40/40 [==============================] - 41s 1s/step - loss: 1.0747 - sparse_categorical_accuracy: 0.8572 - val_loss: 1.0744 - val_sparse_categorical_accuracy: 0.8625\n",
      "Epoch 64/100\n",
      "40/40 [==============================] - 40s 1s/step - loss: 1.0742 - sparse_categorical_accuracy: 0.8608 - val_loss: 1.0741 - val_sparse_categorical_accuracy: 0.8700\n",
      "Epoch 65/100\n",
      "40/40 [==============================] - 42s 1s/step - loss: 1.0736 - sparse_categorical_accuracy: 0.8655 - val_loss: 1.0732 - val_sparse_categorical_accuracy: 0.8810\n",
      "Epoch 66/100\n",
      "40/40 [==============================] - 44s 1s/step - loss: 1.0729 - sparse_categorical_accuracy: 0.8740 - val_loss: 1.0729 - val_sparse_categorical_accuracy: 0.8840\n",
      "Epoch 67/100\n",
      "40/40 [==============================] - 43s 1s/step - loss: 1.0726 - sparse_categorical_accuracy: 0.8817 - val_loss: 1.0724 - val_sparse_categorical_accuracy: 0.8890\n",
      "Epoch 68/100\n",
      "40/40 [==============================] - 42s 1s/step - loss: 1.0721 - sparse_categorical_accuracy: 0.8864 - val_loss: 1.0718 - val_sparse_categorical_accuracy: 0.8930\n",
      "Epoch 69/100\n",
      "40/40 [==============================] - 42s 1s/step - loss: 1.0715 - sparse_categorical_accuracy: 0.8916 - val_loss: 1.0712 - val_sparse_categorical_accuracy: 0.9030\n",
      "Epoch 70/100\n",
      "40/40 [==============================] - 40s 1s/step - loss: 1.0711 - sparse_categorical_accuracy: 0.8940 - val_loss: 1.0709 - val_sparse_categorical_accuracy: 0.9065\n",
      "Epoch 71/100\n",
      "40/40 [==============================] - 40s 1s/step - loss: 1.0703 - sparse_categorical_accuracy: 0.9090 - val_loss: 1.0704 - val_sparse_categorical_accuracy: 0.9140\n",
      "Epoch 72/100\n",
      "40/40 [==============================] - 41s 1s/step - loss: 1.0698 - sparse_categorical_accuracy: 0.9077 - val_loss: 1.0698 - val_sparse_categorical_accuracy: 0.9168\n",
      "Epoch 73/100\n",
      "40/40 [==============================] - 40s 1s/step - loss: 1.0693 - sparse_categorical_accuracy: 0.9165 - val_loss: 1.0693 - val_sparse_categorical_accuracy: 0.9190\n",
      "Epoch 74/100\n",
      "40/40 [==============================] - 41s 1s/step - loss: 1.0687 - sparse_categorical_accuracy: 0.9137 - val_loss: 1.0685 - val_sparse_categorical_accuracy: 0.9280\n",
      "Epoch 75/100\n",
      "40/40 [==============================] - 40s 1s/step - loss: 1.0681 - sparse_categorical_accuracy: 0.9224 - val_loss: 1.0680 - val_sparse_categorical_accuracy: 0.9273\n",
      "Epoch 76/100\n",
      "40/40 [==============================] - 41s 1s/step - loss: 1.0678 - sparse_categorical_accuracy: 0.9196 - val_loss: 1.0677 - val_sparse_categorical_accuracy: 0.9275\n",
      "Epoch 77/100\n",
      "40/40 [==============================] - 42s 1s/step - loss: 1.0674 - sparse_categorical_accuracy: 0.9240 - val_loss: 1.0672 - val_sparse_categorical_accuracy: 0.9302\n",
      "Epoch 78/100\n",
      "40/40 [==============================] - 42s 1s/step - loss: 1.0664 - sparse_categorical_accuracy: 0.9247 - val_loss: 1.0665 - val_sparse_categorical_accuracy: 0.9375\n",
      "Epoch 79/100\n",
      "40/40 [==============================] - 41s 1s/step - loss: 1.0659 - sparse_categorical_accuracy: 0.9337 - val_loss: 1.0659 - val_sparse_categorical_accuracy: 0.9327\n",
      "Epoch 80/100\n",
      "40/40 [==============================] - 41s 1s/step - loss: 1.0656 - sparse_categorical_accuracy: 0.9348 - val_loss: 1.0656 - val_sparse_categorical_accuracy: 0.9310\n",
      "Epoch 81/100\n",
      "40/40 [==============================] - 41s 1s/step - loss: 1.0651 - sparse_categorical_accuracy: 0.9407 - val_loss: 1.0648 - val_sparse_categorical_accuracy: 0.9470\n",
      "Epoch 82/100\n",
      "40/40 [==============================] - 40s 1s/step - loss: 1.0642 - sparse_categorical_accuracy: 0.9410 - val_loss: 1.0642 - val_sparse_categorical_accuracy: 0.9427\n",
      "Epoch 83/100\n",
      "40/40 [==============================] - 41s 1s/step - loss: 1.0639 - sparse_categorical_accuracy: 0.9416 - val_loss: 1.0641 - val_sparse_categorical_accuracy: 0.9427\n",
      "Epoch 84/100\n",
      "40/40 [==============================] - 41s 1s/step - loss: 1.0636 - sparse_categorical_accuracy: 0.9408 - val_loss: 1.0634 - val_sparse_categorical_accuracy: 0.9445\n",
      "Epoch 85/100\n",
      "40/40 [==============================] - 41s 1s/step - loss: 1.0628 - sparse_categorical_accuracy: 0.9472 - val_loss: 1.0630 - val_sparse_categorical_accuracy: 0.9477\n",
      "Epoch 86/100\n",
      "40/40 [==============================] - 41s 1s/step - loss: 1.0622 - sparse_categorical_accuracy: 0.9485 - val_loss: 1.0621 - val_sparse_categorical_accuracy: 0.9498\n",
      "Epoch 87/100\n",
      "40/40 [==============================] - 40s 1s/step - loss: 1.0619 - sparse_categorical_accuracy: 0.9471 - val_loss: 1.0616 - val_sparse_categorical_accuracy: 0.9515\n",
      "Epoch 88/100\n",
      "40/40 [==============================] - 41s 1s/step - loss: 1.0611 - sparse_categorical_accuracy: 0.9543 - val_loss: 1.0612 - val_sparse_categorical_accuracy: 0.9580\n",
      "Epoch 89/100\n",
      "40/40 [==============================] - 40s 1s/step - loss: 1.0605 - sparse_categorical_accuracy: 0.9549 - val_loss: 1.0607 - val_sparse_categorical_accuracy: 0.9615\n",
      "Epoch 90/100\n",
      "40/40 [==============================] - 41s 1s/step - loss: 1.0599 - sparse_categorical_accuracy: 0.9612 - val_loss: 1.0601 - val_sparse_categorical_accuracy: 0.9557\n",
      "Epoch 91/100\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "40/40 [==============================] - 40s 997ms/step - loss: 1.0593 - sparse_categorical_accuracy: 0.9604 - val_loss: 1.0594 - val_sparse_categorical_accuracy: 0.9620\n",
      "Epoch 92/100\n",
      "40/40 [==============================] - 41s 1s/step - loss: 1.0588 - sparse_categorical_accuracy: 0.9625 - val_loss: 1.0588 - val_sparse_categorical_accuracy: 0.9690\n",
      "Epoch 93/100\n",
      "40/40 [==============================] - 41s 1s/step - loss: 1.0584 - sparse_categorical_accuracy: 0.9667 - val_loss: 1.0581 - val_sparse_categorical_accuracy: 0.9725\n",
      "Epoch 94/100\n",
      "40/40 [==============================] - 40s 1s/step - loss: 1.0577 - sparse_categorical_accuracy: 0.9687 - val_loss: 1.0573 - val_sparse_categorical_accuracy: 0.9663\n",
      "Epoch 95/100\n",
      "40/40 [==============================] - 40s 1s/step - loss: 1.0571 - sparse_categorical_accuracy: 0.9670 - val_loss: 1.0572 - val_sparse_categorical_accuracy: 0.9712\n",
      "Epoch 96/100\n",
      "40/40 [==============================] - 42s 1s/step - loss: 1.0564 - sparse_categorical_accuracy: 0.9698 - val_loss: 1.0565 - val_sparse_categorical_accuracy: 0.9775\n",
      "Epoch 97/100\n",
      "40/40 [==============================] - 41s 1s/step - loss: 1.0559 - sparse_categorical_accuracy: 0.9731 - val_loss: 1.0559 - val_sparse_categorical_accuracy: 0.9740\n",
      "Epoch 98/100\n",
      "40/40 [==============================] - 40s 1s/step - loss: 1.0554 - sparse_categorical_accuracy: 0.9735 - val_loss: 1.0553 - val_sparse_categorical_accuracy: 0.9790\n",
      "Epoch 99/100\n",
      "40/40 [==============================] - 41s 1s/step - loss: 1.0547 - sparse_categorical_accuracy: 0.9761 - val_loss: 1.0550 - val_sparse_categorical_accuracy: 0.9808\n",
      "Epoch 100/100\n",
      "40/40 [==============================] - 40s 996ms/step - loss: 1.0543 - sparse_categorical_accuracy: 0.9753 - val_loss: 1.0542 - val_sparse_categorical_accuracy: 0.9780\n",
      "Done!\n",
      "\n",
      "\n"
     ]
    }
   ],
   "source": [
    "#fit model\n",
    "print(\"Starting Training...\")\n",
    "history = model.fit(\n",
    "        train_generator,\n",
    "        steps_per_epoch=40,\n",
    "        epochs=epochs,\n",
    "        validation_data=test_generator,\n",
    "        validation_steps=40,\n",
    "        callbacks =[earlystopping],\n",
    "        shuffle=True)\n",
    "print(\"Done!\\n\\n\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "2fac1525",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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Qxk+fDidO3emUqVKPPzwwwwcOJCxY8cCut/ZJSv3tVChQqSkpHDu3Llr1nGUkhsn8PLyokaNGsyfPz9T+fz586lbt66Losr9jDH069ePH374gYULFxIdHZ1pe3R0NIUKFcp031NSUliyZInu+w1q0qQJW7ZsYePGjRmvmjVr0rVrVzZu3Ejx4sV1r52kXr16V0xpsHv3bqKiogD9XDtbUlISbm6ZP+rc3d0zhoLrfmePrNzXGjVq4OnpmalObGwsW7duvfl7f1PdkSXD5aHgn332mdm+fbsZMGCA8fPzMwcPHnR1aLlWnz59TFBQkFm8eLGJjY3NeCUlJWXUee2110xQUJD54YcfzJYtW8yDDz6oIZxO8vfRUsboXjvL6tWrjYeHh3n11VfNnj17zLRp04yvr6/56quvMuroXjtP9+7dTZEiRTKGgv/www8mJCTEDBs2LKOO7rdjLly4YDZs2GA2bNhgADNu3DizYcOGjClQsnJfe/fubYoWLWr++OMPs379etO4cWMNBc9pPvroIxMVFWW8vLxM9erVM4Ysi2OAq76mTJmSUcdut5sRI0aYQoUKGavVau6++26zZcsW1wWdh/wzudG9dp5ff/3VVKxY0VitVlO2bFnz8ccfZ9que+088fHxpn///iYyMtJ4e3ub4sWLm+eff94kJydn1NH9dsyiRYuu+ju6e/fuxpis3deLFy+afv36meDgYOPj42Puu+8+ExMTc9OxWYwx5ubafkRERERyDvW5ERERkTxFyY2IiIjkKUpuREREJE9RciMiIiJ5ipIbERERyVOU3IiIiEieouRGRERE8hQlNyJy21u8eDEWi4Xz58+7OhQRcQIlNyIiIpKnKLkRERGRPEXJjYi4nDGGN954g+LFi+Pj40OVKlX47rvvgP8/Mpo1axZVqlTB29ub2rVrs2XLlkzH+P7776lQoQJWq5VixYrx9ttvZ9qenJzMsGHDiIiIwGq1UqpUKT777LNMddatW0fNmjXx9fWlbt26V6zeLSK5g5IbEXG5F154gSlTpjBhwgS2bdvGwIEDeeihh1iyZElGnaFDh/LWW2+xZs0aQkNDadOmDampqUB6UtKxY0c6d+7Mli1bePnll3nxxRf5/PPPM/bv1q0b06dP5/3332fHjh1MnDgRf3//THE8//zzvP3226xduxYPDw8eeeSRW3L9IuJcWjhTRFwqMTGRkJAQFi5cSJ06dTLKe/XqRVJSEo8//jiNGjVi+vTpdOrUCYCzZ89StGhRPv/8czp27EjXrl05deoU8+bNy9h/2LBhzJo1i23btrF7927KlCnD/Pnzadq06RUxLF68mEaNGvHHH3/QpEkTAGbPnk2rVq24ePEi3t7e2XwXRMSZ1HIjIi61fft2Ll26xD333IO/v3/Ga+rUqezbty+j3t8Tn+DgYMqUKcOOHTsA2LFjB/Xq1ct03Hr16rFnzx5sNhsbN27E3d2dBg0a/GsslStXzvg6PDwcgJMnT970NYrIreXh6gBE5PZmt9sBmDVrFkWKFMm0zWq1Zkpw/slisQDpfXYuf33Z3xulfXx8shSLp6fnFce+HJ+I5B5quRERlypfvjxWq5WYmBhKliyZ6RUREZFRb+XKlRlfnzt3jt27d1O2bNmMYyxbtizTcZcvX07p0qVxd3enUqVK2O32TH14RCTvUsuNiLhUQEAAQ4YMYeDAgdjtdu666y7i4+NZvnw5/v7+REVFATBq1CgKFChAWFgYzz//PCEhIdx///0ADB48mFq1avHKK6/QqVMnVqxYwYcffsj48eMBKFasGN27d+eRRx7h/fffp0qVKhw6dIiTJ0/SsWNHV126iGQTJTci4nKvvPIKoaGhjB07lv3795MvXz6qV6/Oc889l/FY6LXXXqN///7s2bOHKlWq8Msvv+Dl5QVA9erV+fbbb3nppZd45ZVXCA8PZ9SoUfTo0SPjHBMmTOC5556jb9++nDlzhsjISJ577jlXXK6IZDONlhKRHO3ySKZz586RL18+V4cjIrmA+tyIiIhInqLkRkRERPIUPZYSERGRPEUtNyIiIpKnKLkRERGRPEXJjYiIiOQpSm5EREQkT1FyIyIiInmKkhsRERHJU5TciIiISJ6i5EZERETyFCU3IiIikqf8FwY4KUb9oMdDAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "def plot_hist(hist):\n",
    "    plt.plot(hist.history['sparse_categorical_accuracy'])\n",
    "    plt.plot(hist.history['val_sparse_categorical_accuracy'])\n",
    "    plt.title('CategoricalCrossentropy')\n",
    "    plt.ylabel('sparse_categorical_accuracy')\n",
    "    plt.xlabel('epoch')\n",
    "    plt.legend(['train', 'validation'], loc='upper left')\n",
    "    plt.show()\n",
    "\n",
    "plot_hist(history)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "b70d5548",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "44/44 [==============================] - 14s 325ms/step - loss: 1.0541 - sparse_categorical_accuracy: 0.9772\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "[1.054124116897583, 0.9771689772605896]"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.evaluate(test_generator)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "89ec44b3",
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Example Predicted Set:\n",
      "\n",
      "\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<Figure size 1200x2000 with 60 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "path = '/home/vlad/Desktop/allimages/1/'\n",
    "dirList = os.listdir(path)\n",
    "plt.figure(figsize=(12,20))\n",
    "count = 0\n",
    "imageList = []\n",
    "label = 'Low Confidence'\n",
    "\n",
    "one = 0\n",
    "two = 0\n",
    "three = 0\n",
    "\n",
    "for i in dirList[:60]:\n",
    "    #print(i)\n",
    "    image = Image.open(path + str(i))\n",
    "    #plt.figure()\n",
    "    #plt.imshow(image)\n",
    "    #print(f\"Original size : {image.size}\")\n",
    "        \n",
    "    image = image.resize((img_width, img_height))\n",
    "    #print(f\"New size : {image.size}\")\n",
    "\n",
    "    x = np.asarray(image)   \n",
    "    x = np.expand_dims(x, axis=0)\n",
    "    #print(x/255.0)\n",
    "    images = np.vstack([x])\n",
    "    #print(images[0][50])\n",
    "    classes = model.predict(\n",
    "    images,\n",
    "    batch_size=1,\n",
    "    verbose='off',\n",
    "    steps=None,\n",
    "    callbacks=None,\n",
    "    max_queue_size=10,\n",
    "    workers=1,\n",
    "    use_multiprocessing=True)\n",
    "\n",
    "    classifier = \"  Low Confidence\"\n",
    "    \n",
    "    score = float(classes[0][0])\n",
    "    #print(f\"   This image is {100 * (score):.2f}% far-field.\")\n",
    "    if score > 0.66:\n",
    "        classifier = f\"{100 * (score):.2f}%  wedge\"\n",
    "        label = ['wedge']\n",
    "        one += 1\n",
    "   \n",
    "    score = float(classes[0][1])\n",
    "    #print(f\"   This image is {100 * (score):.2f}% wedge.\")\n",
    "    if score > 0.66:\n",
    "        classifier = f\"{100 * (score):.2f}%  near-field\"\n",
    "        label = ['near-field']\n",
    "        two += 1\n",
    "    \n",
    "    score = float(classes[0][2])\n",
    "    #print(f\"   This image is {100 * (score):.2f}% near-field.\")\n",
    "    if score > 0.66:\n",
    "        classifier= f\"{100 * (score):.2f}%  far-field\"\n",
    "        label = ['far-field']\n",
    "        three += 1\n",
    "    \n",
    "    imageList += label\n",
    "    \n",
    "    \n",
    "    if classifier == \"  Low Confidence\":\n",
    "        classifier = f\"{100 * (classes[0].max()):.2f}% Low Conf.\"\n",
    "        \n",
    "    if count < 60:\n",
    "        ax=plt.subplot(10,6,count+1)\n",
    "        plt.imshow(image, cmap='gray')\n",
    "        plt.title(str(i[0:6] + \"   \" + classifier), fontsize = 7)\n",
    "        count += 1\n",
    "        plt.axis('off')\n",
    "print(\"Example Predicted Set:\\n\\n\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "710e0402",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.9998305\n",
      "[1.6947555e-04 9.9983048e-01 8.9609662e-13]\n"
     ]
    }
   ],
   "source": [
    "print(classes[0].max())\n",
    "print(classes[0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "9ae0b482",
   "metadata": {},
   "outputs": [],
   "source": [
    "nameList = []\n",
    "index = 0\n",
    "while index < len(dirList[:]):\n",
    "    if dirList[index][0:3] == 'B12':\n",
    "        nameList += ['near-field']\n",
    "    elif dirList[index][0:3] == 'B11':\n",
    "        nameList += ['wedge']\n",
    "    elif dirList[index][0:4] == 'ATW1':\n",
    "        nameList += ['near-field']\n",
    "    elif dirList[index][0:5] == 'FE2_C':\n",
    "        nameList += ['wedge']\n",
    "    elif dirList[index][0:5] == 'FE2_S':\n",
    "        nameList += ['near-field']\n",
    "    elif dirList[index][0:4] == 'BTW1':\n",
    "        nameList += ['near-field']\n",
    "    elif dirList[index][0:3] == 'A12':\n",
    "        nameList += ['near-field']\n",
    "    elif dirList[index][0:4] == 'ATW2':\n",
    "        nameList += ['far-field']\n",
    "    else:\n",
    "        nameList += ['far-field']\n",
    "    index += 1    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "9862d436",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[114,  53, 140],\n",
       "       [ 15, 795,   9],\n",
       "       [  0,   2, 331]])"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.metrics import confusion_matrix\n",
    "\n",
    "y_true = nameList\n",
    "y_pred = imageList\n",
    "confusion_matrix(y_true, y_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "20a54210",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "High Confidence Images: 92.32 % of total (greater than 66% confidence value).\n",
      "\n",
      "\n",
      "441/480 wedge predictions: 0-\n",
      "789/850 near-field predictions: 1-\n",
      "117/129 far-field predictions: 2-\n",
      "\n",
      "\n",
      "1240 values correct out of 1459: 84.99% model accuracy\n"
     ]
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 640x480 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "from sklearn.metrics import ConfusionMatrixDisplay\n",
    "cm = confusion_matrix(y_true, y_pred)\n",
    "\n",
    "farTrue=cm[0][0]+cm[1][0]+cm[2][0]\n",
    "nearTrue=cm[0][1]+cm[1][1]+cm[2][1]\n",
    "wedgeTrue=cm[0][2]++cm[1][2]+cm[2][2]\n",
    "\n",
    "correctValue = cm[0][0]+cm[1][1]+cm[2][2]\n",
    "\n",
    "print(f\"High Confidence Images: {100*(one+two+three)/(index):.2f} % of total (greater than 66% confidence value).\")\n",
    "print(\"\\n\")\n",
    "\n",
    "print(f\"{(one)}/{(wedgeTrue)} wedge predictions: 0-\")\n",
    "print(f\"{(two)}/{(nearTrue)} near-field predictions: 1-\")\n",
    "print(f\"{(three)}/{(farTrue)} far-field predictions: 2-\")\n",
    "\n",
    "\n",
    "print(\"\\n\")\n",
    "print(f\"{correctValue} values correct out of {index}: {100 * (correctValue/index):.2f}% model accuracy\")\n",
    "cm_display = ConfusionMatrixDisplay(cm).plot()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "dfde8dbf",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.16"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}