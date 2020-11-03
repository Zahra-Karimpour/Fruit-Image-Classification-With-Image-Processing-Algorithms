import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import sys
import scipy
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from  sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
# import sklearn

Classes2Indices = {'Apple Braeburn': 0, 'Apple Granny Smith': 1, 'Apricot': 2,
               'Banana Lady Finger': 3, 'Cantaloupe 1': 4, 'Clementine': 5,
               'Corn': 6, 'Grapefruit White': 7, 'Kaki': 8, 'Kumquats': 9,
               'Onion Red': 10, 'Peach': 11, 'Pepper Orange': 12,
               'Pomegranate': 13, 'Tomato 4': 14}

Indices2Classes = dict((value,key) for key,value in Classes2Indices.items())

def scorer(preds, labels):
  assert preds.shape[0] == labels.shape[0]
  # x = labels.shape[0]

  # counter = 0
  # for i in range(x):
  #   if preds[i]==labels[i]:
  #     counter+=1
  return np.mean(preds==labels)
  # return (counter/x)





if __name__ == "__main__":
  # Loading data
  TrainData = np.load('./extracted_features/Features.npy')
  TestData = np.load('./extracted_features/Features.npy')

  ## If training requires shuffling, uncomment this line
  np.random.shuffle(TrainData)

  #  Calculate  absolute of logarithms
  TrainData[:,4] = np.abs(TrainData[:,4])
  TestData[:,4] = np.abs(TestData[:,4])


  # FEATURES: 0:dominantColors, 1:max_height, 2:max_width, 3:hw_proportion,
  # FEATURES: 4:log_hw_proportion, 5:skeleton_size, 6:sum_canny, 7:sum_cannyRGB, 
  # FEATURES: 8:mean_w, 9:mean_h, 10:dominantSaturation, 11:sum_canny1, 12:labels

  my_feature_set = (0,1,2,5,6,7,10)
  print(my_feature_set)

  # Processing data
  TrainFeatures, TrainLabels = TrainData[:,my_feature_set] , TrainData[:,-1]
  TestFeatures, TestLabels  = TestData[:,my_feature_set], TestData[:,-1]


  RFClassifier = RandomForestClassifier()
  RFClassifier.fit(TrainFeatures, TrainLabels)
  max_acc= RFClassifier.score(TestFeatures,TestLabels)
  max_x = -1
  for i in range(10):
    x = np.random.randint(89,240)

    RFClassifier = RandomForestClassifier(n_estimators=x)
    RFClassifier.fit(TrainFeatures, TrainLabels)
    scoreTest = RFClassifier.score(TestFeatures,TestLabels)
    if max_acc<scoreTest:
      max_acc = scoreTest
      max_x=x
    

  ETClassifier = ExtraTreesClassifier()
  print(ETClassifier)
  ETClassifier.fit(TrainFeatures, TrainLabels)
  max_acc= ETClassifier.score(TestFeatures,TestLabels)
  max_x = -1
  for i in range(10):
    x = np.random.randint(87,275)
    ETClassifier = ExtraTreesClassifier(n_estimators=x)
    ETClassifier.fit(TrainFeatures, TrainLabels)
    scoreTest = ETClassifier.score(TestFeatures,TestLabels)
    if max_acc<scoreTest:
      max_acc = scoreTest
      max_x=x
    


  print(max_x, max_acc)





