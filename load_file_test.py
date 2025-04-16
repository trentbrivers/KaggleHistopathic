import cv2 as cv
import pandas as pd
import os
import numpy as np

original_shape = (96,96,3)
coming_in = np.loadtxt('my_array.csv', delimiter = ",")
coming_in = coming_in.reshape(original_shape)
print(coming_in)