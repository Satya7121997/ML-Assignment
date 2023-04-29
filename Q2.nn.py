#!/usr/bin/env python
# coding: utf-8

# In[28]:

from tensorflow import keras
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

get_ipython().run_line_magic('matplotlib', 'inline')
import os
import cv2
from keras.layers import Dense, Flatten
train_path = 'C:\\Users\\satya\\Documents\\train-20230429T100501Z-001\\'
val_path = 'C:\\Users\\satya\\Documents\\val-20230429T100436Z-001\\'
data_dir = train_path
img_size = (32, 32)
images = []
labels = []
for label in range(10):
    folder_path = os.path.join(data_dir, 'train', str(label))
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if file_path.endswith(('.tiff','.bmp')):
 
             img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
             img = cv2.resize(img, img_size)
 
             images.append(img)
             labels.append(label)
images = np.array(images)
labels = np.array(labels)
np.save('x_train.npy', images)
np.save('y_train.npy', labels)
data_dir_val = val_path
img_size_val = (32, 32)
images_val = []
labels_val = []
for label in range(10):
    folder_path = os.path.join(data_dir_val, 'val\\', str(label))

 
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if file_path.endswith(('.tiff','.bmp')):
 
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, img_size_val)
 
            images_val.append(img)
            labels_val.append(label)
images_val = np.array(images_val)
labels_val = np.array(labels_val)
np.save('x_test.npy', images_val)
np.save('y_test.npy', labels_val)
x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')
x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')
print(len(x_train))
print(len(x_test))
x_train[0].shape
x_train[0]
plt.matshow(x_train[0])
plt.matshow(x_train[999])
print(x_train.shape)
print(x_test.shape)
y_train
y_test
plt.matshow(x_test[150])
model = keras.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(10, input_shape=(1024,),activation = 'sigmoid')
])
model.compile(optimizer='adam',
 loss='sparse_categorical_crossentropy',
 metrics=['accuracy']
 )
model.fit(x_train, y_train,epochs= 10, validation_data=(x_test, y_test))
x_train_scaled = x_train/255
x_test_scaled = x_test/255
model.fit(x_train_scaled, y_train,epochs= 10, validation_data=(x_test_scaled, y_test))
model.evaluate(x_test_scaled,y_test)
plt.matshow(x_test[0])
y_predicted = model.predict(x_test_scaled)
y_predicted[0]
print('Predicted Value is ',np.argmax(y_predicted[0]))
plt.matshow(x_test[88])
print('Predicted Value is ',np.argmax(y_predicted[88]))
plt.matshow(x_test[177])
print('Predicted Value is ',np.argmax(y_predicted[177]))
y_predicted_labels=[np.argmax(i) for i in y_predicted]
print(y_predicted_labels, len(y_predicted_labels))
conf_mat = tf.math.confusion_matrix(labels=y_test, predictions=y_predicted_labels)
conf_mat
import seaborn as sn
plt.figure(figsize = (10,10))
sn.heatmap(conf_mat,annot=True,fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
model2 = keras.Sequential([
 keras.layers.Flatten(),
 keras.layers.Dense(1024,input_shape=(1024,), activation='relu'),
 keras.layers.Dense(10, activation='softmax')
])
model2.compile(optimizer='adam',
 loss='sparse_categorical_crossentropy',
 metrics=['accuracy']
 )
history = model2.fit(x_train_scaled, y_train,epochs= 10, validation_data=(x_test_scaled, y_test))
y_predicted = model2.predict(x_test_scaled)
y_predicted[0]
y_predicted_labels=[np.argmax(i) for i in y_predicted]
print(y_predicted_labels, len(y_predicted_labels))
conf_mat = tf.math.confusion_matrix(labels=y_test, predictions=y_predicted_labels)
conf_mat
plt.figure(figsize = (10,10))
sn.heatmap(conf_mat,annot=True,fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


# In[ ]:




