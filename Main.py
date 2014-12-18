'''
Created on 06.05.2014.

@author: d.day
'''

import Person
import cv2
import Image
import numpy as np
import GPA
import matplotlib.pyplot as plt
import PCA as pca



if __name__ == '__main__':
    pass

Persons=[]

#lista svih osoba
for i in range(1,15):
    Persons.append(Person.Person(i))

#Procustres normalizacija sekutica na isti koordinatni sistem za svaki sekutic posebno
for inc in range(0,8):

    npc=np.array([Persons[0].Incisors[inc].xy,Persons[1].Incisors[inc].xy])
    for p in range(2,len(Persons)):
        npc=np.concatenate((npc,[Persons[p].Incisors[inc].xy]))

    (mean_shape0,shapes)=GPA.generalized_procrustes_analysis(npc)

    '''
    plt.scatter(mean_shape0[:,0],mean_shape0[:,1])
    print (npc)
    poly=plt.Polygon(mean_shape0)
    plt.gca().add_patch(poly)
    plt.show()
    '''

    for p in range(0,len(Persons)):
        Persons[p].Incisors[inc].normXY=shapes[p]


#spaja sve primere prvog sekutica u matricu gde su u jednom redu spojeni vektori koordinata lendmarkova (x,y) a u koloni razliciti primeri 14x80
data=np.append(Persons[0].Incisors[0].normXY[:,0],Persons[0].Incisors[0].normXY[:,1])
for p in range(1,14):
    data=np.vstack((data,np.append(Persons[p].Incisors[0].normXY[:,0],Persons[p].Incisors[0].normXY[:,1])))

#PCA sa 3 komponente
eigenvalues, eigenvectors, mu=pca.pcaD(data,3)

#formatiranje izlaza PCA u pogodan oblik za plot
tEVectors=np.array((eigenvectors[0:40,:],eigenvectors[40:80,:])).T
mean_shape=np.array((mu[0:40],mu[40:80])).T

'''
#rekonstrukcija nekog zuba preko opsteg modela
b0=np.dot(tEVectors[0].T,(Persons[0].Incisors[5].normXY-mean_shape))
b1=np.dot(tEVectors[1].T,(Persons[0].Incisors[5].normXY-mean_shape))
b2=np.dot(tEVectors[2].T,(Persons[0].Incisors[5].normXY-mean_shape))

'''
#b parametri oblika zuba
#b0 regulise sirinu i oblost zuba (-1 do 0,7)
#b1 regulise da li je spic na levoj ili desnoj strani (-1 do 1)
#b2 regulise da li je krunica veca od korena (-1 do 1)
#b=np.zeros((3))
b0=-0.5*np.sqrt(eigenvalues[0])
b1=-1*np.sqrt(eigenvalues[1])
b2=-1*np.sqrt(eigenvalues[2])


x=mean_shape+np.dot(tEVectors[0],b0)+np.dot(tEVectors[1],b1)+np.dot(tEVectors[2],b2)

plt.scatter(x[:,0],x[:,1])
#plt.scatter(tEVectors[:,0],tEVectors[:,1])
#plt.scatter(ty[:,0],ty[:,1])
#plt.scatter(eigenvectors[0:2,0], eigenvectors[0:2,1])
plt.axis('scaled')
plt.show()

'''
# 2x4 PCA plot
f, axarr = plt.subplots(2, 4)
a=0
for j in range(0,2):
    for i in range(0,4):
        axarr[j,i].plot(incPCA[a][:,1],incPCA[a][:,0],'o', markersize=3, color='blue', alpha=0.8)
        axarr[j,i].set_title('PCA '+str(a))
        axarr[j,i].axis('scaled')
        a=a+1
plt.show()'''

f, axarr = plt.subplots(2, 4)
a=0
for j in range(0,2):
    for i in range(0,4):
        axarr[j,i].scatter(Persons[0].Incisors[a].normXY[:,0],Persons[0].Incisors[a].normXY[:,1])
        axarr[j,i].set_title('Incisor '+str(a))
        axarr[j,i].axis('scaled')
        a=a+1
plt.show()

path='img/'
#load data images
imgs=Image.loadImages(path)
#do adaptive histogram equalization
imgs=Image.clahe(imgs)
#draw imgs and landmarks
Image.drawProNormalization(imgs, Persons)

