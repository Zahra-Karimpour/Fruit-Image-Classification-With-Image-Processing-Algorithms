import os
import numpy as np
import cv2
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import sys
import scipy
from skimage.morphology import skeletonize, thin, opening, erosion
from skimage import feature
from scipy import ndimage as ndi
from algorithms import _extract_foreground_single_image, extract_foreground, extract_color_from_hue_histogram, maximum_height_and_width, \
                        maximum_height_and_width, extracting_skeleton, derivatives, derivatives_2, mean_height_and_width, derivatives3

ROOT_DIR = './Data/Train/'
_DEBUG = False


# To Add: Sum of MASK, 
class FruitDataset(object):
  def __init__(self, root_dir='./Data/Train/'):
    
    self.root_dir = root_dir
    print(root_dir)
    TrainingDir = list(sorted(os.listdir(root_dir)))
    if _DEBUG:
      TrainingDir = root_dir
    self.TrainClassesList = np.array([(i, os.path.join(root_dir, i)) for i in TrainingDir  if i[0]!='.'])

    self.TrainClassesNames = np.array([x[0] for x in self.TrainClassesList])
    

    self.ClassNameToIndexTrain = {self.TrainClassesNames[i]:i for i in range(len(self.TrainClassesNames))}
    self.IndexToClassNameTrain = {i:self.TrainClassesNames[i] for i in range(len(self.TrainClassesNames))}

    # Pairs: (class name, addr_to_image)
    self.TrainImages = np.array([os.path.join(x[1], y) for x in self.TrainClassesList for y in list(sorted(os.listdir(x[1])))])
    self.TrainLabels = np.array([x[0] for x in self.TrainClassesList for y in list(sorted(os.listdir(x[1])))])


  def __getitem__(self, idx):
    _addrs = self.TrainImages[idx]
    _labels = self.TrainLabels[idx]
  
    labels = np.array([self.ClassNameToIndexTrain[i] for i in _labels]).reshape((-1,1))
    print(labels.shape)
    images = np.array([np.array(Image.open(i).convert("RGB")).astype(np.float32)  for i in _addrs if i.split('/')[-1][0] != '.'])

    masks, foregrounds = extract_foreground(images)
    


    dominantColors, dominantSaturation = extract_color_from_hue_histogram(foregrounds)
    max_height, max_width, hw_proportion, log_hw_proportion = maximum_height_and_width(masks)
    mean_w, mean_h  = mean_height_and_width(masks)
    print("YO")
    _, skeleton_size = extracting_skeleton(masks)

    sum_canny = derivatives(foregrounds, masks)
    sum_cannyRGB = derivatives_2(foregrounds, masks)
    sum_canny1 = derivatives3(foregrounds,masks)

    extracted_features = np.concatenate((dominantColors, max_height, max_width, hw_proportion, log_hw_proportion, skeleton_size, sum_canny, sum_cannyRGB, mean_w, mean_h, dominantSaturation, sum_canny1, labels), axis=-1)
    
    return extracted_features

  def __len__(self):
      return len(self.TrainImages)





if  __name__ == "__main__":
  my_data = FruitDataset(root_dir= './Data/Test/')
  print(len(my_data))
  features = my_data[range(len(my_data))]
  print(len(my_data))

  np.save('./extracted_features/Features.npy', features)

  my_data = FruitDataset(root_dir= './Data/Train/')
  print(len(my_data))
  features = my_data[range(len(my_data))]
  print(len(my_data))

  np.save('./extracted_features/Features.npy', features)