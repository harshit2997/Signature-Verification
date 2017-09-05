import numpy as np
import cv2
from scipy import fftpack
from scipy import stats
from scipy.spatial import distance
import glob,os,collections
from scipy.stats import entropy 
import math
import pickle as pk

filenames = list()
test_filenames=list()

dctCoefficients = list()
test_dctCoefficients=list()

Y = list()
test_Y=list()

#create list of reference and test filenames

os.chdir(r'./TrainB')

for file in glob.glob("*.PNG"):
    filenames.append(file)
    Y.append(1)

os.chdir(r'..')    

os.chdir(r'./TestB/1')
for file in glob.glob("*.PNG"):
    test_filenames.append(file)
    test_Y.append(1) 
    
os.chdir(r'..')    
os.chdir(r'./0')
for file in glob.glob("*.PNG"):
    test_filenames.append(file)
    test_Y.append(0)

#preprocessing and feature extraction of reference images
    
os.chdir(r'../..')   
os.chdir(r'./TrainB')
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
    im2=cv2.resize(img,dim,interpolation=cv2.INTER_AREA)   #resize with larger dimension set to 100
    
    ret,binarised = cv2.threshold(im2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)  #Otsu binarisation
    if col>row:
        resized=cv2.copyMakeBorder(binarised,top=0,bottom=100-binarised.shape[0],left=0,right=0,borderType= cv2.BORDER_CONSTANT, value=255) #apply margins
    else:
        resized=cv2.copyMakeBorder(binarised,top=0,bottom=0,left=0,right=100-binarised.shape[1],borderType= cv2.BORDER_CONSTANT, value=255) #apply margins
    
    kernel = np.ones((3,3),np.uint8)
    resized = cv2.erode(resized,kernel,iterations = 1)  #erosion
    
    resized=np.array(resized,dtype=np.float)
    
    dct=fftpack.dct(fftpack.dct(resized.T, norm='ortho').T, norm='ortho')   #apply DCT to get coefficients
    dct_copy = dct[:5,:10]  #take first 5 X 10 DCT coefficients as features
    dctCoefficients.append(dct_copy.ravel())
   
#preprocessing and feature extraction of test images

os.chdir(r'..')       
os.chdir(r'./TestB/C')
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
        resized=cv2.copyMakeBorder(binarised,top=0,bottom=100-binarised.shape[0],left=0,right=0,borderType= cv2.BORDER_CONSTANT, value=255) #apply margins
    else:
        resized=cv2.copyMakeBorder(binarised,top=0,bottom=0,left=0,right=100-binarised.shape[1],borderType= cv2.BORDER_CONSTANT, value=255) #apply margins
    
    kernel = np.ones((5,5),np.uint8)
    resized = cv2.erode(resized,kernel,iterations = 1)  #erosion
    
    resized=np.array(resized,dtype=np.float)
    
    dct=fftpack.dct(fftpack.dct(resized.T, norm='ortho').T, norm='ortho')   #apply DCT to get coefficients
    dct_copy = dct[:5,:10]  #take first 5 X 10 DCT coefficients as features
    test_dctCoefficients.append(dct_copy.ravel())

os.chdir(r'../..') 

dwk=[]
lentr=len(dctCoefficients)

for x in range(lentr):
    dctCoefficients[x]=list(dctCoefficients[x])

#create within known distribution

dwk=distance.pdist(dctCoefficients,'euclidean')
dwk=list(dwk)

alpha=0.03

tp=0
tn=0
fp=0
fn=0
tot=0

for samp in test_dctCoefficients:
    cl=0
    dqk=[]
    for i in range(lentr):
        dqk.append(distance.euclidean(dctCoefficients[i],samp)) #create questioned vs known distribution
    p_ks= stats.ks_2samp(np.array(dwk),np.array(dqk))[1]    #apply KS test of similarity of distributions

#apply alpha for classification
#cl=1 for genuine , cl=0 for forgery

    if p_ks>alpha:
        cl=1
    
    if cl==1 and test_Y[tot]==1:
        tp+=1
    elif cl==1 and test_Y[tot]==0:
        fp+=1
    elif cl==0 and test_Y[tot]==1:
        fn+=1
    elif cl==0 and test_Y[tot]==0:
        tn+=1

    tot+=1

print str(alpha)
print ({str(alpha):[(float(fn)/float(tp+fn)),(float(fp)/float(fp+tn))]})    #print FRR and FAR




