# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 02:10:19 2021

@author: Romil
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from tqdm import tqdm
#%%
y= [None for x in range(100)]
file1 = open("frame (1).txt","r")
x=file1.read()
file1.close()
s=0
T=[]
for i in x:
    T.append(' ')
    if i ==' ':
        s=s+1
    else:
        T[s]=T[s]+i
#%%
R=[1,2,3,4]
for i in range(0,4):
    R[i]=float(T[i])

#%%
y=[]
#%%
y.append(R)
#%%
x.split()
x=(x)
#%%
X=[]
y=[]
t='frame ('
e=').jpg'
r=').txt'
h=64
w=64
for i in tqdm(range(1,271)):
   img = cv2.imread(t+str(i)+e,1)
   try:
       img=cv2.resize(img,(100,100))
       imgs=np.array(img).flatten()
       X.append(imgs).flatten()
   except Exception as R:
            pass
   file1 = open(t+str(i)+r,"r")
   x=file1.read()
   file1.close()
   s=0
   T=[]
   for i in x:
       T.append(' ')
       if i ==' ':
           s=s+1
       else:
           T[s]=T[s]+i
   R=[0,2,3,4,5]
   for i in range(1,5):
       R[i]=float(T[i])
   y.append(R)
#%%
X = np.array(X)
y = np.array(y)
X=X.astype('float32')

X/= 255

#%%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#%%
from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(X_train,y_train)
#%%
accuracy =reg.score(X_test,y_test)
#%%
print('Accuracy of Linear=',accuracy*100)
#%%
from sklearn.ensemble import RandomForestRegressor
lin_reg = RandomForestRegressor(n_estimators=10,random_state =0)
lin_reg.fit(X, y)
acc=lin_reg.score(X_test,y_test)
#%%
print('Accuracy of Linear=',acc*100)
#%%
#%%
tsts=cv2.imread('frame (1).jpg',1)
tst1=cv2.resize(tsts,(100,100))
tst=np.array(tst1).flatten()
print(tst)
yhat=lin_reg.predict([tst])
#%%
print(yhat)
xs=int(yhat[0][1]*1920)+1
ys=int(yhat[0][2]*1080)+1
ws=int(yhat[0][3]*1920)+1
hs=int(yhat[0][4]*1080)+1

imgt=cv2.rectangle(tsts,(xs,ys),(ws,hs),(255,0,0),2)
#%%
cv2.imshow("test",imgt)
cv2.waitKey(0)
cv2.imwrite('test.png',imgt)
cv2.destroyAllWindows()


