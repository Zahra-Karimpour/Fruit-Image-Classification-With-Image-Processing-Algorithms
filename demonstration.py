import os
import numpy as np
import cv2
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import sys
import scipy
from skimage.morphology import skeletonize, thin, opening, erosion, remove_small_holes
from skimage import feature
from scipy import ndimage as ndi
from  scipy.ndimage.morphology  import binary_fill_holes
from algorithms import _extract_foreground_single_image, extract_foreground, extract_color_from_hue_histogram, maximum_height_and_width, \
                        maximum_height_and_width, extracting_skeleton, derivatives, derivatives_2, mean_height_and_width, derivatives3




ROOT_DIR = './Data/Train/'
_DEBUG = False

TrainingDir = list(sorted(os.listdir(ROOT_DIR)))


TrainClassesList = [(i, os.path.join(ROOT_DIR, i)) for i in TrainingDir  if i[0]!='.']
TrainClassesNames = [x[0] for x in TrainClassesList]

ClassNameToIndexTrain = {TrainClassesNames[i]:i for i in range(len(TrainClassesNames))}
IndexToClassNameTrain = {i:TrainClassesNames[i] for i in range(len(TrainClassesNames))}


TrainImages = np.array([os.path.join(x[1], y) for x in TrainClassesList for y in list(sorted(os.listdir(x[1])))])
TrainLabels = np.array([x[0] for x in TrainClassesList for y in list(sorted(os.listdir(x[1])))])

addr0 = 'Data/Train/Kaki/129_100.jpg'
addr1 = 'Data/Train/Tomato 4/52_100.jpg'
addr2 = 'Data/Train/Pepper Orange/6_100.jpg'

addrs = [addr0,addr1]

images = np.array([np.array(Image.open(i).convert("RGB")).astype(np.float32)  for i in addrs])

masks, foregrounds = extract_foreground(images)
mhs, mws, _, _ = maximum_height_and_width(masks)
fig,axs = plt.subplots(1, 6)








for i in  range(images.shape[0]):
  image = images[i]

  imgcopy = np.copy(image)
  lowerBound = (170,170,170)
  upperBound = (255,255,255)
  thresh = cv2.inRange(imgcopy, lowerBound, upperBound)
  thresh = 255 - thresh
  mask = np.tile(thresh[:,:,np.newaxis],[1,1,3])/255

  axs[3*i].axis('off')
  axs[3*i].imshow(mask*image/255,cmap='gray')
  axs[3*i].title.set_text('Thresh')


  axs[3*i+2].axis('off')
  axs[3*i+2].imshow(erosion(erosion(erosion(erosion(mask))))*image/255,cmap='gray')
  axs[3*i+2].title.set_text('Eroded')

  mask[:,:,2] = mask[:,:,1] =  mask[:,:,0] = binary_fill_holes(mask[:,:,0].astype('int32'))
  
  mask = erosion(mask)
  mask = erosion(mask)

  axs[3*i+1].axis('off')
  axs[3*i+1].imshow(mask*image/255,cmap='gray')
  axs[3*i+1].title.set_text('Filled')
  foreground =  image*mask/255.





plt.savefig('./mask_hole.png', dpi= 1200)