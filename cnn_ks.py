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
import pickle as pk
import random

wpath='weights_700_50ep.hdf5' #path of stored weights

#building and compiling the neural network

opt = SGD(lr=0.01)
model= arch.build(width=100, height=100, depth=1, classes=2,    
    weightsPath=wpath)
model.compile(loss="categorical_crossentropy", optimizer=opt,
    metrics=["accuracy"])

filenames = []
test_filenames=[]

train_features = []
test_features=[]

data=[]
test_data=[]

Y = []
test_Y=[]

#listing of filenames of test and reference images

os.chdir(r'./TrainE')

for file in glob.glob("*.png"):
    filenames.append(file)
    Y.append(1)

os.chdir(r'..')    

os.chdir(r'./TestE/1')
for file in glob.glob("*.png"):
    test_filenames.append(file)
    test_Y.append(1) 
    
os.chdir(r'..')    
os.chdir(r'./0')
for file in glob.glob("*.png"):
    test_filenames.append(file)
    test_Y.append(0)

#preprocessing of reference images
    
os.chdir(r'../..')   
os.chdir(r'./TrainE')
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
    im2=cv2.resize(img,dim,interpolation=cv2.INTER_AREA)  #resize with larger dimension set to 100
    
    ret,binarised = cv2.threshold(im2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)  #Otsu binarisation
    if col>row:
        temp=int((100-binarised.shape[0])/2)
        resized=cv2.copyMakeBorder(binarised,top=100-binarised.shape[0]-temp,bottom=temp,left=0,right=0,borderType= cv2.BORDER_CONSTANT, value=255)  #adding margins
    else:
        temp=int((100-binarised.shape[1])/2)
        resized=cv2.copyMakeBorder(binarised,top=0,bottom=0,left=100-binarised.shape[1]-temp,right=temp,borderType= cv2.BORDER_CONSTANT, value=255)  #adding margins 
    
    kernel = np.ones((3,3),np.uint8)
    resized = cv2.erode(resized,kernel,iterations = 1)  #erosion    
    
    resized=np.array(resized,dtype=np.float)

    resized=resized/255.0   
    data.append(resized)


#preprocessing of test images   
    
os.chdir(r'..')       
os.chdir(r'./TestE/C')
for i in range(len(test_filenames)):
    name=test_filenames[i]

    img=cv2.imread(name,0)
    row,col= img.shape[:2]
    ratio=float(row)/float(col)
    if col>row:
        dim=(100,int(ratio*100))
    else:
        ratio=1/ratio
        dim=(int(ratio*100),100)
    im2=cv2.resize(img,dim,interpolation=cv2.INTER_AREA)    #resize with larger dimension set to 100
    
    ret,binarised = cv2.threshold(im2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)  #Otsu binarisation
    if col>row:
        temp=int((100-binarised.shape[0])/2)
        resized=cv2.copyMakeBorder(binarised,top=100-binarised.shape[0]-temp,bottom=temp,left=0,right=0,borderType= cv2.BORDER_CONSTANT, value=255) #adding margins
    else:
        temp=int((100-binarised.shape[1])/2)
        resized=cv2.copyMakeBorder(binarised,top=0,bottom=0,left=100-binarised.shape[1]-temp,right=temp,borderType= cv2.BORDER_CONSTANT, value=255) #adding margins
    
    kernel = np.ones((3,3),np.uint8)
    resized = cv2.erode(resized,kernel,iterations = 1) #erosion
    
    resized=np.array(resized,dtype=np.float)
    resized=resized/255.0
    test_data.append(resized)
 

data=np.array(data)
test_data=np.array(test_data)
data = data[:, :, :, np.newaxis]
test_data = test_data[:, :, :, np.newaxis]

#obtain output of fully connected layer to use as feature vector

for i in range (len(data)):
    get_fc_layer_output = K.function([model.layers[0].input],
                                  [model.layers[11].output])
    fc_layer_output = get_fc_layer_output([data[np.newaxis,i]])[0]
    train_features.append(fc_layer_output[0])    

for i in range (len(test_data)):
    get_fc_layer_output = K.function([model.layers[0].input],
                                  [model.layers[11].output])
    fc_layer_output = get_fc_layer_output([test_data[np.newaxis,i]])[0]
    test_features.append(fc_layer_output[0])  

train_features=np.array(train_features)
test_features=np.array(test_features)


len_test=len(test_features)

dwk=[]

lentr=len(train_features)

for x in range(lentr):
    train_features[x]=list(train_features[x])

#create within known distribution

dwk=distance.pdist(train_features,'euclidean')
dwk=list(dwk)


tp=0
tn=0
fp=0
fn=0
tot=0


alpha=0.11
pred=[]
ind=[]
tpc=0
tnc=0
puc=0
nuc=0

for samp in test_features:
    cl=0
    dqk=[]
    for i in range(lentr):
        dqk.append(distance.euclidean(train_features[i],samp))   #create questioned vs known distribution
    p_ks= stats.ks_2samp(np.array(dwk),np.array(dqk))[1]    #apply KS test of similarity of distributions

#apply alpha and beta for classification
#cl=1 for genuine , cl=0 for forgery , cl=-1 for uncertainity
    if p_ks>alpha+0.1:     
        cl=1
    elif p_ks<=alpha+0.1 and p_ks>=alpha-0.1:
    	cl=-1
    
    pred.append(cl)
    if cl==-1 and test_Y[tot]==1 and puc<1:
        puc+=1
        ind.append(tot)

    elif cl==-1 and test_Y[tot]==0 and nuc<1:
        nuc+=1
        ind.append(tot)

    elif cl==1 and test_Y[tot]==1:
        tp+=1
        if tpc<4:
            tpc+=1
            ind.append(tot)
    elif cl==1 and test_Y[tot]==0:
        fp+=1        
    elif cl==0 and test_Y[tot]==1:
        fn+=1
    elif cl==0 and test_Y[tot]==0:
        tn+=1
        if tnc<4:
            tnc+=1
            ind.append(tot)        


    tot+=1

print str(alpha)
print ({str(alpha):[(float(fn)/float(tp+fn)),(float(fp)/float(fp+tn))]})  #print FRR and FAR

#show some results

print "Displaying some results...."


for i in ind:
    if test_Y[i]:
        s1="Genuine"
    else:
        s1="Forgery"

    if pred[i]==1:
        s2="Genuine"
    elif pred[i]==0:
        s2="Forgery"
    else:
        s2="Uncertainity" 

    print "Actual : "+s1+" | Predicted : "+s2
    img=cv2.imread(test_filenames[i],1)
    cv2.imshow("Image",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



os.chdir(r'../..')





