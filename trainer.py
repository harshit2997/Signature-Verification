from network import arch
from sklearn.cross_validation import train_test_split
from sklearn import datasets
from keras.optimizers import SGD
from keras.utils import np_utils
import tensorflow as tf
import numpy as np
import argparse
from keras import backend as K
import theano
import struct
import cv2
from scipy import stats
from scipy.spatial import distance
import glob,os,collections
from scipy.stats import entropy 
import math


filenames=list()
Y = list()
data=[]

#create list of filenames

os.chdir(r'./TestB/1')
for file in glob.glob("*.PNG"):
    filenames.append(file)
    Y.append(1) 
    
os.chdir(r'..')    
os.chdir(r'./0')
for file in glob.glob("*.PNG"):
    filenames.append(file)
    Y.append(0)
    
os.chdir(r'../..')   
os.chdir(r'./TestB/C')

#preprocessing of images

for i in range(len(filenames)):
    name=filenames[i]
    img=cv2.imread(name,0)
    row,col= img.shape[:2]
    ratio=float(row)/float(col)
    if col>row:
        dim=(100,int(ratio*100))
    else:
        ratio=1/ratio
        dim=(int(ratio*100),100)
    im2=cv2.resize(img,dim,interpolation=cv2.INTER_AREA)    #resizing with larger dimension set to 100
    
    ret,binarised = cv2.threshold(im2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)  #Otsu binarisation 
    if col>row:
        temp=int((100-binarised.shape[0])/2)
        resized=cv2.copyMakeBorder(binarised,top=100-binarised.shape[0]-temp,bottom=temp,left=0,right=0,borderType= cv2.BORDER_CONSTANT, value=255) #add margins
    else:
        temp=int((100-binarised.shape[1])/2)
        resized=cv2.copyMakeBorder(binarised,top=0,bottom=0,left=100-binarised.shape[1]-temp,right=temp,borderType= cv2.BORDER_CONSTANT, value=255) #add margins
    
    
    kernel = np.ones((3,3),np.uint8)
    resized = cv2.erode(resized,kernel,iterations = 1)  #erosion
    
    resized=np.array(resized,dtype=np.float)

    resized=resized/255.0
    data.append(resized)

os.chdir(r'../..')  

data=np.array(data)
Y=np.array(Y)

data = data[:, :, :, np.newaxis]
(trainData, testData, trainLabels, testLabels) = train_test_split(  
    data, Y.astype("int"), test_size=0.25)  #split data in the ratio 3:1 for training and testing data

#convert class tags (0/1) into 2 membere vectors

trainLabels = np_utils.to_categorical(trainLabels, 2)
testLabels = np_utils.to_categorical(testLabels, 2)


print("[INFO] training...")

for temp in [50]:
    opt = SGD(lr=0.01)
    model = arch.build(width=100, height=100, depth=1, classes=2,weightsPath=None)  #build model architecture
    model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])  #compile model
    model.fit(trainData, trainLabels, batch_size=2, nb_epoch=int(temp),verbose=1)   #fit model to training data with no. of epoch=50 and batch size=2

    print("[INFO] evaluating...")
    (loss, accuracy) = model.evaluate(testData, testLabels,
            batch_size=2, verbose=1)
    print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))


    print("[INFO] dumping weights to file...")
    name='weights_700_'+str(temp)+'ep.hdf5'
    model.save_weights(name, overwrite=True)    #saving weights
