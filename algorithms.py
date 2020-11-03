import os
import numpy as np
import cv2
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import sys
import scipy
from skimage.morphology import skeletonize, thin, opening, erosion, binary_erosion
from skimage import feature
from scipy import ndimage as ndi
from  scipy.ndimage.morphology  import binary_fill_holes






def _extract_foreground_single_image(image):
  imgcopy = np.copy(image)
  lowerBound = (230,230,230)
  upperBound = (255,255,255)
  thresh = cv2.inRange(imgcopy, lowerBound, upperBound)
  thresh = 255 - thresh
  mask = np.tile(thresh[:,:,np.newaxis],[1,1,3])/255

  mask[:,:,2] = mask[:,:,1] =  mask[:,:,0] = binary_fill_holes(mask[:,:,0].astype('int32'))
  
  # Erode the mask to make sure it captures only the fruit
  mask = erosion(mask)
  mask = erosion(mask)
  foreground =  image*mask/255.

  return mask, foreground


def extract_foreground(images):
  if images.ndim == 4: 
    imagesCopy = np.copy(images)
    foregrounds = np.zeros_like(imagesCopy)
    masks = np.zeros_like(imagesCopy)
    for  i in range(imagesCopy.shape[0]):
      m, f = _extract_foreground_single_image(imagesCopy[i])
      masks[i] = m
      foregrounds[i] = f
      

  return masks, foregrounds

def extract_color_from_hue_histogram(foregrounds):
  foregrounds = np.copy(foregrounds)
  if foregrounds.ndim == 3:
    foregrounds.reshape((1,foregrounds.shape))
  if foregrounds.ndim == 4:
    print(foregrounds.shape)
    dominant_colors = np.zeros([foregrounds.shape[0], 1])
    dominant_saturation = np.zeros([foregrounds.shape[0], 255])
    i = -1


    for foreground in foregrounds:
      i+=1
      foregroundsHSV = cv2.cvtColor((foreground*255).astype('uint8'),cv2.COLOR_RGB2HLS)
      histr = cv2.calcHist([foregroundsHSV[:,:,0]],[0],None, [256], [1,256])  
      histrSaturation = cv2.calcHist([foregroundsHSV[:,:,1]],[0],None, [256], [1,256])  

      dominant_colors[i,0] = histr.argmax()
      dominant_saturation[i,0] = histrSaturation.argmax()

  return dominant_colors, dominant_saturation

def maximum_height_and_width(masks): # Add proportion and log of proportion
  max_w = np.zeros([masks.shape[0], 1])
  max_h = np.zeros([masks.shape[0], 1])

  for x in range(masks.shape[0]):
    _mask  =  masks[x]
    mask =  np.mean(_mask,-1)
    max_width_x = 0
    max_height_x = 0

    a,b = mask.shape
    assert(a==b)
    for i in range(30,a-30):
      temp_height = 0
      temp_width = 0
      for  j in range(0,b): 
        if mask[i,j]>0:
          temp_height +=1
        else:
          if max_height_x <= temp_height:
            max_height_x = temp_height
            temp_height = 0

        if mask[j,i]>0:
            temp_width +=1
        else:
          if max_width_x <=temp_width:
            max_width_x = temp_width
            temp_width = 0

      if max_width_x <= temp_width:
        max_width_x = temp_width
      if max_height_x <= temp_height:
        max_height_x = temp_height
      
      max_w[x,0]  =  max_width_x
      max_h[x,0] = max_height_x

  return max_h, max_w, max_h/max_w, np.log(max_h/max_w)

def extracting_skeleton(masks,foregrounds=  None):
  masks = np.copy(masks)
  skeletons = np.zeros(masks.shape[:-1])
  sum_skeletons = np.zeros((masks.shape[0], 1))
  for i in range(masks.shape[0]):
    mask  = np.mean(masks[i],-1)
    mask  =  opening(mask)
    sklt = skeletonize(mask)
    skeletons[i] = sklt

  _sum_skeletons = np.sum(np.reshape(skeletons, [skeletons.shape[0],-1]),-1)
  sum_skeletons  =  _sum_skeletons.reshape((-1,1))
  return skeletons, sum_skeletons

def derivatives(foregrounds, masks):
  foregrounds = np.copy(foregrounds)

  if foregrounds.ndim == 3:
    foreground.reshape((1, foreground.shape))
  if masks.ndim == 3:
    masks.reshape((1, masks.shape))
  
  sum_derivatives = np.zeros([masks.shape[0], 1])

  for i in range(foregrounds.shape[0]):
    foreground = (foregrounds[i]*255).astype('uint8')
    mask  = erosion(np.mean(masks[i],-1))

    ## Sum/Max between channels
    # edges0 = feature.canny(foregrounds[:,:,0], sigma = 1 )
    # edges1 = feature.canny(foregrounds[:,:,1], sigma = 1 )
    # edges2 = feature.canny(foregrounds[:,:,2], sigma = 1 )

    # sum_derivatives[i] = np.sum(edges0*mask)+np.sum(edges1*mask)+np.sum(edges2*mask)
    
    # SUM of VALUE in  HSV
    foregroundHSV = cv2.cvtColor(foreground ,cv2.COLOR_RGB2HSV)
    edges = feature.canny(foregroundHSV[:,:,-1], sigma = 2 )
    sum_derivatives[i] = np.sum(edges * erosion(mask))

  return sum_derivatives

def derivatives3(foregrounds, masks):
  foregrounds = np.copy(foregrounds)

  if foregrounds.ndim == 3:
    foreground.reshape((1, foreground.shape))
  if masks.ndim == 3:
    masks.reshape((1, masks.shape))
  
  sum_derivatives = np.zeros([masks.shape[0], 1])

  for i in range(foregrounds.shape[0]):
    foreground = (foregrounds[i]*255).astype('uint8')
    mask  = erosion(np.mean(masks[i],-1))

    ## Sum/Max between channels
    # edges0 = feature.canny(foregrounds[:,:,0], sigma = 1 )
    # edges1 = feature.canny(foregrounds[:,:,1], sigma = 1 )
    # edges2 = feature.canny(foregrounds[:,:,2], sigma = 1 )

    # sum_derivatives[i] = np.sum(edges0*mask)+np.sum(edges1*mask)+np.sum(edges2*mask)
    
    # SUM of VALUE in  HSV
    foregroundHSV = cv2.cvtColor(foreground ,cv2.COLOR_RGB2HSV)
    edges = feature.canny(foregroundHSV[:,:,-1], sigma = 1 )
    sum_derivatives[i] = np.sum(edges * erosion(mask))

  return sum_derivatives

def derivatives_2(foregrounds, masks):
  foregrounds = np.copy(foregrounds)

  if foregrounds.ndim == 3:
    foreground.reshape((1, foreground.shape))
  if masks.ndim == 3:
    masks.reshape((1, masks.shape))
  
  sum_derivatives = np.zeros([masks.shape[0], 1])

  for i in range(foregrounds.shape[0]):
    foreground = (foregrounds[i]*255).astype('uint8')
    mask  = erosion(np.mean(masks[i],-1))

    ## Sum/Max between channels
    edges0 = feature.canny(foreground[:,:,0], sigma = 1 )
    edges1 = feature.canny(foreground[:,:,1], sigma = 1 )
    edges2 = feature.canny(foreground[:,:,2], sigma = 1 )

    sum_derivatives[i,0] = np.sum(edges0*mask)+np.sum(edges1*mask)+np.sum(edges2*mask)
    
  return sum_derivatives

def mean_height_and_width(masks): # Add proportion and log of proportion
  mean_w = np.zeros([masks.shape[0], 1])
  mean_h = np.zeros([masks.shape[0], 1])

  for x in range(masks.shape[0]):
    _mask  =  masks[x]
    mask = np.mean(_mask,-1)
    assert mask.shape == (100,100)
    mw = np.sum(np.mean(mask, axis=0))
    mh = np.sum(np.mean(mask, axis=1))
    mean_w[x,0] = mw
    mean_h[x,0] = mh

  return mean_w, mean_h
