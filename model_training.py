import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.regularizers import l2

import cv2 as cv
import numpy as np
import os

def list_files_in_directory(folder_path):
  """Lists all files in the specified directory.

  Args:
    folder_path: The path to the directory.

  Returns:
    A list of strings, where each string is the name of a file in the directory.
    Returns an empty list if the directory does not exist or is empty.
  """
  try:
    files = os.listdir(folder_path)
    return [f for f in files if os.path.isfile(os.path.join(folder_path, f))]
  except FileNotFoundError:
    return []

path = r"E:\Data\histopathologic-cancer-detection\train"

training_files = list_files_in_directory(path)

#Settings for me:
limiter = 50000 #restrict the total number of samples from the whole batch
model_name = "iteration_10k.keras"

if limiter:
	X_train = np.array([cv.imread(os.path.join(path,file)) for file in training_files[:limiter]]) #create an np array of images
	y_train = np.loadtxt('proper_ordered_labels.csv', delimiter=',')[:limiter] #a np array consisting of the labels
else:
	X_train = np.array([cv.imread(os.path.join(path,file)) for file in training_files])
	y_train = np.loadtxt('proper_ordered_labels.csv', delimiter=',') #a np array consisting of the labels

#Validate inputs
#print(X_train[0], X_train.shape, type(X_train))
#print(y_train[0], type(y_train[0]), y_train.shape, type(y_train))

model = Sequential([
	Conv2D(96, (3,3), activation = "relu", input_shape = (96,96,3)),
	Flatten(),
	keras.layers.Dense(64, activation='relu', kernel_regularizer=l2(0.001)), 
	keras.layers.Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
	keras.layers.Dense(1, activation='sigmoid') #Final binary output

])

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train, epochs=3, batch_size=16)

model.save(os.path.join(r"E:\Data\KaggleHistopathic", model_name))