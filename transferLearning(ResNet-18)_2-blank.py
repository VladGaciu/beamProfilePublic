#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 11:14:04 2024

@author: vladGACIU
"""
#https://medium.com/@kirudang/deep-learning-computer-vision-using-transfer-learning-ResNet-18-3-Channel-in-pytorch-skin-cancer-8d5b158893c5

# Set up CUDA in OS
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# Import libabries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import seaborn as sn
import pandas as pd
import torchvision
from torchvision import *
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import torchvision.transforms as T
from torchvision import datasets, models, transforms
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import time
import copy

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
print("all libraries imported")

# Check version of Pytorch
print(torch. __version__)

# Setup device agnostic code
device = "cuda"
print(device)

# Find out if a GPU is available
use_cuda = torch.cuda.is_available()
print(use_cuda)

# Set up path for data after downloading
train_dir = '/home/vlad/Desktop/ResNet-18-3-Channel/Training/'
test_dir = '/home/vlad/Desktop/ResNet-18-3-Channel/Test/'


train_classa_dir = "/home/vlad/Desktop/ResNet-18-3-Channel/Training/NF/"
train_classb_dir = "/home/vlad/Desktop/ResNet-18-3-Channel/Training/FF/"
train_classc_dir = "/home/vlad/Desktop/ResNet-18-3-Channel/Training/WEDGE/"
train_classd_dir = "/home/vlad/Desktop/ResNet-18-3-Channel/Training/NOISE/"

test_classa_dir = '/home/vlad/Desktop/ResNet-18-3-Channel/Test/NF/'
test_classb_dir = '/home/vlad/Desktop/ResNet-18-3-Channel/Test/FF/'
test_classc_dir = '/home/vlad/Desktop/ResNet-18-3-Channel/Test/WEDGE/'
test_classd_dir = '/home/vlad/Desktop/ResNet-18-3-Channel/Test/NOISE/'

# Display NF image for reference
# white_torch = torchvision.io.read_image('/home/vlad/Desktop/ResNet-18-3-Channel/Training/NF/A12_2021-08-20 10_06_02.957.png')
# print("... this is near-field training image")
# T.ToPILImage()(white_torch)

# # Display FF image for reference
# wh = torchvision.io.read_image('/home/vlad/Desktop/ResNet-18-3-Channel/Training/FF/ATW2_2021-08-20 10_07_47.753.png')
# print("... this is far-field training image")
# #T.ToPILImage()(wh)

# # Display WEDGE image for reference
# wh = torchvision.io.read_image('/home/vlad/Desktop/ResNet-18-3-Channel/Training/WEDGE/B11_2021-08-20 12_03_47.165.png')
# print("... this is wedge training image")
# #T.ToPILImage()(wh)

# # Display NOISE image for reference
# wh = torchvision.io.read_image('/home/vlad/Desktop/ResNet-18-3-Channel/Training/NOISE/A12_2021-08-20 18_01_25.953.png')
# print("... this is NOISE training image")
# #T.ToPILImage()(wh)


# Create transform function
transforms_train = transforms.Compose([
    transforms.Resize((224, 224)),   #must same as here
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(), # data augmentation
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalization
])
transforms_test = transforms.Compose([
    transforms.Resize((224, 224)),   #must same as here
     transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Apply for training and test data
train_dataset = datasets.ImageFolder(train_dir, transforms_train)
test_dataset = datasets.ImageFolder(test_dir, transforms_test)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=12, shuffle=True, num_workers=0)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=12, shuffle=False, num_workers=0)

print('Train dataset size:', len(train_dataset))
print('Test dataset size:', len(test_dataset))
class_names = train_dataset.classes
print('Class names:', class_names)

model = models.resnet18(pretrained=True)
model

num_features = model.fc.in_features 
print('Number of features from pre-trained model', num_features)

# Add a fully-connected layer for classification
model.fc = nn.Linear(num_features, 4)
model = model.to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.00001, momentum=0.9)

# Set the random seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)

#### Train model
train_loss=[]
train_accuary=[]
test_loss=[]
test_accuary=[]

num_epochs = 30   #(set no of epochs)
start_time = time.time() #(for showing time)
# Start loop
for epoch in range(num_epochs): #(loop for every epoch)
    print("Epoch {} running".format(epoch)) #(printing message)
    """ Training Phase """
    model.train()    #(training model)
    running_loss = 0.   #(set loss 0)
    running_corrects = 0 
    # load a batch data of images
    for i, (inputs, labels) in enumerate(train_dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device) 
        # forward inputs and get output
        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        # get loss value and update the network weights
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data).item()
    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects / len(train_dataset) * 100.
    # Append result
    train_loss.append(epoch_loss)
    train_accuary.append(epoch_acc)
    # Print progress
    print('[Train #{}] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.format(epoch+1, epoch_loss, epoch_acc, time.time() -start_time))
    """ Testing Phase """
    model.eval()
    with torch.no_grad():
        running_loss = 0.
        running_corrects = 0
        for inputs, labels in test_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data).item()
        epoch_loss = running_loss / len(test_dataset)
        epoch_acc = running_corrects / len(test_dataset) * 100.
        # Append result
        test_loss.append(epoch_loss)
        test_accuary.append(epoch_acc)
        # Print progress
        print('[Test #{}] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.format(epoch+1, epoch_loss, epoch_acc, time.time()- start_time))


save_path = 'custom-classifier_resnet_18_final.pth'
torch.save(model.state_dict(), save_path)

# Plot
plt.figure(figsize=(6,6))
plt.plot(np.arange(1,num_epochs+1), train_accuary,'-o')
plt.plot(np.arange(1,num_epochs+1), test_accuary,'-o')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train','Test'])
plt.title('Train vs Test Accuracy over time')
plt.show()

#%%

# Apply for training and test data
val_dir = '/home/vlad/Desktop/ResNet-18-3-Channel/Validation/'


val_dataset = datasets.ImageFolder(val_dir, transforms_test)

val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=12, shuffle=False, num_workers=0)

# Get data to check on the performance of each label
y_pred = []
y_true = []

num_epochs = 30   #(set no of epochs)
start_time = time.time() #(for showing time)
# Start loop
for epoch in range(num_epochs): #(loop for every epoch)
    model.eval()
    with torch.no_grad():
        for inputs, labels in val_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs) # Feed Network
            outputs = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
            y_pred.extend(outputs) # Save Prediction
            labels = labels.data.cpu().numpy()
            y_true.extend(labels) # Save Truth
            
# Visualization and result
# constant for classes
classes = val_dataset.classes
# Build confusion matrix
print("Accuracy on Validation set: ",accuracy_score(y_true, y_pred))
print('Confusion matrix: \n', confusion_matrix(y_true, y_pred))
print('Classification report: \n', classification_report(y_true, y_pred))
# Plot
cf_matrix = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cf_matrix, index = [i for i in classes], columns = [i for i in classes])
plt.figure(figsize = (7,7))
plt.title("Confusion matrix for beam profile classification ")
sn.heatmap(df_cm, annot=True)
