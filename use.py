import argparse
import os, pdb, sys, glob, time
from random import shuffle
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
from matplotlib import pyplot as plt

# import neccesary libraries for defining the optimizers
import torch.optim as optim
import os
from tqdm import tqdm

import efficientnet.tfkeras as efn

import random
import cv2
from keras import backend as K
from keras.preprocessing import image

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

import os
from glob import glob

from tensorflow.keras.preprocessing import image

import cv2

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.models import load_model

# from tensorflow.keras.applications import DenseNet121
import tensorflow as tf
import tensorflow.keras.layers as L
# import tensorflow.keras.layers as Layers

# model import
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.nasnet import NASNetLarge
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.xception import Xception
from keras.applications.densenet import DenseNet201

# model
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from keras.models import Sequential
from keras import optimizers, callbacks, regularizers

IMAGE_SIZE=[256, 256]

base_model = efn.EfficientNetB1(
                    input_shape = (*IMAGE_SIZE, 3), 
                    include_top = False, 
                    # weights = None
                    weights='imagenet'
                    )
model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.5))
model.add(Dense(512))
model.add(Dropout(0.5))
model.add(Dense(14, activation = 'sigmoid'))
model.compile(
    optimizer=tf.keras.optimizers.Adam( learning_rate=1e-4, amsgrad=False), 
    loss = 'binary_crossentropy',
    metrics = ['binary_accuracy']
)
model.load_weights('C:/Users/hb/Desktop/code/2.TF_CZ/Weight/SOTA.h5')
pathes = glob('C:/Users/hb/Desktop/data/Generated Image/PGGAN1/*.png')

df = pd.DataFrame(columns =['Cardiomegaly','Emphysema','Effusion','Hernia','Infiltration','Mass','Nodule','Atelectasis','Pneumothorax','Pleural_Thickening','Pneumonia','Fibrosis','Edema','Consolidation','FilePath'])

# Preprocessing
image_generator = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization= True, 
        shear_range=0.1,
        zoom_range=0.15,
        rotation_range=5,
        width_shift_range=0.1,
        height_shift_range=0.05,
        horizontal_flip=False, 
        vertical_flip = False, 
        fill_mode = 'reflect')

for i in tqdm(range(len(pathes))):

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    # img = np.array(image.load_img(pathes[i], target_size=(256, 256))).astype('float64')
    img = cv2.imread(pathes[i])
    img = np.expand_dims(img, axis=0)
    img = image_generator.flow(img)
    pred = model.predict(img)

    id = i
    name = pathes[i].split('\\')[1]

    item = []
    for d in range(14):
        item.append(int(pred[0][d])) 
    item.append(pathes[i])

    df.loc[i] = item
    
df.to_csv('pggan_df.csv')

    
