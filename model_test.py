import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.regularizers import l2

import cv2 as cv
import numpy as np
import os
from tqdm import tqdm

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

path = r"E:\Data\histopathologic-cancer-detection\test"

model = keras.models.load_model(r"E:\Data\KaggleHistopathic\iteration_50k.keras")

test_files = list_files_in_directory(path)

test_images = np.array([cv.imread(os.path.join(path,file)) for file in test_files])
predictions = model.predict(test_images, batch_size = None)
predictions = [1 if p >= 0.5 else 0 for p in predictions]

with open("submission2.csv", "w") as file:
  file.write("id,label\n")
  for i in tqdm(range(len(test_files))):
    current_id = test_files[i][:-4]
    current_label = predictions[i]
    file.write(current_id + "," + str(current_label) + "\n")