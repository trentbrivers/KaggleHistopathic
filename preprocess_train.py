import cv2 as cv
import pandas as pd
import os
import numpy as np
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


#tif image format: img[row][column][B, G, R]

path = r"E:\Data\histopathologic-cancer-detection\train"

labels_path = r"E:\Data\histopathologic-cancer-detection\train_labels.csv"

labels_data = pd.read_csv(labels_path)

files_in_train = list_files_in_directory(path)
#print(files_in_train[0], type(files_in_train[0]))
#Getting all file names works

file_path = os.path.join(path, files_in_train[0])

#test_image = cv.imread(file_path)
#image_string = files_in_train[0][:-4]
#print(test_image)
#print(image_string)
#current_label = labels_data[(labels_data['id'] == image_string)]['label'].values[0]
#print(current_label)
#print(files_in_train[0])
#print(test_image)
#Confirmed that it gets file from path
#save_arr = test_image.ravel()

#np.savetxt("my_array.csv", save_arr, delimiter=",", fmt="%d")
#Test 1 - using np -> csv 
#Works

proper_labels = []

for image in tqdm(files_in_train):
	image_string = image[:-4]
	image_label = int(labels_data[(labels_data['id'] == image_string)]['label'].values[0])
	proper_labels.append(image_label)

proper_labels_np = np.array(proper_labels)
np.savetxt("proper_ordered_labels.csv", proper_labels_np, delimiter = ",", fmt = "%d")
print(proper_labels_np)