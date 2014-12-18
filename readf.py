'''
Created on 05.05.2014.

@author: d.day
'''
import cv2
import numpy as np
import procrustes

if __name__ == '__main__':
    pass

f = open('lm/landmarks1-1.txt', 'r')

Lf=[]

for line in f:
    Lf=Lf+[int(float(line))]

#xy=zip(Lf[0::2], Lf[1::2])

xy=np.empty((len(Lf)/2,2),np.int32)

for x in range(len(Lf)/2):
    xy[x]=Lf[2*x],Lf[2*x+1]


f1 = open('lm/landmarks2-1.txt', 'r')
Lf1=[]

for line in f1:
    Lf1=Lf1+[int(float(line))]


xy1=np.empty((len(Lf1)/2,2),np.int32)

for x in range(len(Lf)/2):
    xy1[x]=Lf1[2*x],Lf1[2*x+1]


[d,Z,t]=procrustes.procrustes(xy,xy1)
Zi=np.int32(Z)
print Zi


img = cv2.imread('img/01.tif')
cv2.namedWindow("mainw",0)   ## create window for display
cv2.resizeWindow("mainw", 1500,900);


cv2.polylines(img,[xy],True,(0,0,255),2)
cv2.polylines(img,[xy1],True,(0,255,0),2)
cv2.polylines(img,[Zi],True,(255,0,0),5)

cv2.imshow('mainw',img)
cv2.waitKey(0)
cv2.destroyAllWindows()