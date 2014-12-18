'''
Created on 06.05.2014.

@author: d.day
'''
import cv2
from os import walk
import numpy as np



def loadImages(path):
    #load all images from radiology folder
    imgs=[]
    (_, _, filenames) = walk(path).next();
    for f in range(len(filenames)):
        imgs.append(cv2.imread(path+str(filenames[f]),0))
    return imgs

def drawProNormalization(imgs,Persons,alfa=0):

    #draw alfa as blue, current landmarks - green, landmarks after normalization red
    for p in range(len(Persons)):
        img = imgs[p]
        for i in range(8):
            #cv2.polylines(img,[alfa[i].xy],True,(0,0,255),2)
            cv2.polylines(img,[Persons[p].Incisors[i].xy],True,(0,255,0),2)
            #cv2.polylines(img,[Persons[p].Incisors[i].normXY],True,(255,0,0),5)

        cv2.namedWindow("mainw",0)   ## create window for display
        cv2.resizeWindow("mainw", 1500,900);
        cv2.imshow('mainw',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def prepareIMG(imgs):

    gray_imgs = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in imgs)



def clahe(imgs):

    imgsCORR=[]
    # create a CLAHE object (cliplimit - contrast clip, tileGridSize - sample size).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(32,32))

    for img in imgs:
        imgsCORR.append (clahe.apply(img))

    return imgsCORR



