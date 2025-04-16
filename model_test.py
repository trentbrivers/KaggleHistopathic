import keras
from keras.models import Sequential
from keras.layers import Dense

import cv2 as cv
import pandas as pd
import os
import numpy as np

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


#tif image format: img[row][column][B, G, R]

path = r"E:\Data\histopathologic-cancer-detection\train"

files_in_train = list_files_in_directory(path)
#print(files_in_train[0], type(files_in_train[0]))
#Getting all file names works

file_path = os.path.join(path, files_in_train[0])

test_image = cv.imread(file_path)