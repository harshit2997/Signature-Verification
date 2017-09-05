from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2, numpy as np


class arch:
    @staticmethod
    def build(width, height, depth, classes, weightsPath=None):  

        '''
        method to define the archtecture of the convolutional neural network

        returns the final model with loaded weigths if a weights path is specified
        '''  
             
        model = Sequential()
        model.add(ZeroPadding2D((1,1),input_shape=(width, height, depth)))  #zero padding layer  
        model.add(Convolution2D(16, 4, 4, activation='relu'))   #4 X 4 convolution with 16 filters and ReLu activation
        model.add(ZeroPadding2D((1,1))) #zero padding layer
        model.add(Convolution2D(16, 4, 4, activation='relu'))   #4 X 4 convolution with 16 filters and ReLu activation
        model.add(MaxPooling2D((2,2), strides=(2,2)))   #2 X 2max pooling layer with stride of 2

        model.add(ZeroPadding2D((1,1))) #zero padding layer
        model.add(Convolution2D(32, 4, 4, activation='relu'))   #4 X 4 convolution with 32 filters and ReLu activation
        model.add(ZeroPadding2D((1,1))) #zero padding layer
        model.add(Convolution2D(32, 4, 4, activation='relu'))   #4 X 4 convolution with 32 filters and ReLu activation
        model.add(MaxPooling2D((2,2), strides=(2,2)))   #2 X 2max pooling layer with stride of 2

        model.add(Flatten())
        model.add(Dense(700, activation='relu'))    #fully connected layer with 700 output features

        model.add(Dense(2, activation='softmax'))   #softmax classification layer

        if weightsPath is not None:
            model.load_weights(weightsPath) #load weights if weights path is specified

        return model

