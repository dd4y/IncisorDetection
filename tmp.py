'''
Created on 15.05.2014.

@author: d.day
'''

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
#from matplotlib.mlab import PCA as mlabPCA
from sklearn.decomposition import PCA



if __name__ == '__main__':
    pass

def mult(matrix1,matrix2):
    # Matrix multiplication
    if len(matrix1[0]) != len(matrix2):
        # Check matrix dimensions
        print 'Matrices must be m*n and n*p to multiply!'
    else:
        # Multiply if correct dimensions
        new_matrix = zero(len(matrix1),len(matrix2[0]))
        for i in range(len(matrix1)):
            for j in range(len(matrix2[0])):
                for k in range(len(matrix2)):
                    new_matrix[i][j] += matrix1[i][k]*matrix2[k][j]
        return new_matrix
def zero(m,n):
    # Create zero matrix
    new_matrix = [[0 for row in range(n)] for col in range(m)]
    return new_matrix


Persons=[]

for i in range(1,15):
    Persons.append(Person.Person(i))

for inc in range(0,8):
    npc=np.array([Persons[0].Incisors[inc].xy,Persons[1].Incisors[inc].xy])

    for p in range(2,len(Persons)):
        npc=np.concatenate((npc,[Persons[p].Incisors[inc].xy]))

    (mean_shape0,shapes)=GPA.generalized_procrustes_analysis(npc)
    '''plt.scatter(mean_shape0[:,0],mean_shape0[:,1])
    poly=plt.Polygon(mean_shape0)
    plt.gca().add_patch(poly)'''
    plt.show()

    for p in range(0,len(Persons)):
        Persons[p].Incisors[inc].normXY=shapes[p]


npc=np.array([Persons[0].Incisors[0].xy,Persons[1].Incisors[0].xy])
for p in range(2,len(Persons)):
    npc=np.concatenate((npc,[Persons[p].Incisors[0].xy]))



incPCA=[]
'''
for inc in range(0,8):
    x=np.array([])
    y=np.array([])
    for p in range(0,14):
        a=Persons[p].Incisors[inc].normXY
        x=np.append(x,(a[:,0]))
        y=np.append(y,a[:,1])
    data=np.vstack((x,y))

    #mean, eigenvectors = cv2.PCACompute(npc, np.mean(npc, axis=0).reshape(1,-1))
    #mlab_pca = mlabPCA(data.T)
    sklearn_pca = PCA(n_components=2)
    incPCA.append(sklearn_pca.fit_transform(data.T))'''


#print(incPCA[0].components_)

#spaja sve primere prvog sekutica u matricu gde su u jednom redu spojeni vektori koordinata lendmarkova (x,y) a u koloni razliciti primeri 14x80
data1=np.append(Persons[0].Incisors[0].normXY[:,0],Persons[0].Incisors[0].normXY[:,1])
for p in range(1,14):
    data1=np.vstack((data1,np.append(Persons[p].Incisors[0].normXY[:,0],Persons[p].Incisors[0].normXY[:,1])))

eigenvalues, eigenvectors, mu=pca.pcaD(data1,3)


tEVectors=np.array((eigenvectors[0:40,:],eigenvectors[40:80,:])).T

ty=np.array((mu[0:40],mu[40:80])).T

'''
#spaja sve primere prvog sekutica u jednu x,y matricu 560x2
x=np.array([])
y=np.array([])
for p in range(0,14):
    a=Persons[p].Incisors[0].normXY
    x=np.append(x,(a[:,0]))
    y=np.append(y,a[:,1])
data=np.vstack((x,y)).T

mu = data.mean(axis=0)
data = data - mu
# data = (data - mu)/data.std(axis=0)  # Uncomment this reproduces mlab.PCA results
eigenvectors, eigenvalues, V = np.linalg.svd(
    data.T, full_matrices=False)
projected_data = np.dot(data, eigenvectors)
sigma = projected_data.std(axis=0).mean()
#print(eigenvectors)
def annotate(ax, name, start, end):
    arrow = ax.annotate(name,
                        xy=end, xycoords='data',
                        xytext=start, textcoords='data',
                        arrowprops=dict(facecolor='red', width=2.0))
    return arrow

fig, ax = plt.subplots()
ax.scatter(data[:,0], data[:,1])
ax.set_aspect('equal')
for axis in eigenvectors:
    annotate(ax, '', mu, mu + sigma * axis)
plt.show()
'''




b0=np.dot(tEVectors[0].T,(Persons[0].Incisors[1].normXY-ty))
b1=np.dot(tEVectors[1].T,(Persons[0].Incisors[5].normXY-ty))

b0=0*np.sqrt(eigenvalues[0])
b1=0*np.sqrt(eigenvalues[1])
b2=1*np.sqrt(eigenvalues[2])

#tt=Persons[0].Incisors[0].normXY[0:4]-ty[0:4]

#b=np.array(mult(tt,tEVectors.T))


#b=np.zeros((40,2))

#b[0]=-2.2*eigenvalues[0]
#b[1]=1.9*eigenvalues[1]

print tEVectors[0].shape
print b1.shape

#b=3*np.sqrt(eigenvalues[0])
#x=ty+np.dot(tEVectors[0],b0)

x=ty+np.dot(tEVectors[0],b0)+np.dot(tEVectors[1],b1)+np.dot(tEVectors[2],b2)

plt.scatter(x[:,0],x[:,1])
#plt.scatter(tEVectors[:,0],tEVectors[:,1])
#plt.scatter(ty[:,0],ty[:,1])
#plt.scatter(eigenvectors[0:2,0], eigenvectors[0:2,1])
plt.axis('scaled')
plt.show()

# 2x4 PCA plot
f, axarr = plt.subplots(2, 4)
a=0
for j in range(0,2):
    for i in range(0,4):
        axarr[j,i].plot(incPCA[a][:,1],incPCA[a][:,0],'o', markersize=3, color='blue', alpha=0.8)
        axarr[j,i].set_title('PCA '+str(a))
        axarr[j,i].axis('scaled')
        a=a+1
plt.show()

f, axarr = plt.subplots(2, 4)
a=0
for j in range(0,2):
    for i in range(0,4):
        axarr[j,i].scatter(Persons[4].Incisors[a].normXY[:,0],Persons[4].Incisors[a].normXY[:,1])
        axarr[j,i].set_title('Incisor '+str(a))
        axarr[j,i].axis('scaled')
        a=a+1
plt.show()

'''
for i in range(0,8):
    plt.scatter(Persons[4].Incisors[i].normXY[:,0],Persons[4].Incisors[i].normXY[:,1])
    poly=plt.Polygon(Persons[4].Incisors[i].normXY)
    plt.gca().add_patch(poly)
    plt.axis('scaled')
    plt.show()

for i in range (0,8):s
    for j in range (1,14):
        [d,Z,t]=procrustes.procrustes(alfa[i].xy,Persons[j].Incisors[i].xy)
        Persons[j].Incisors[i].normXY=np.int32(Z)
x=[]
y=[]
for i in range(len(mean_shape0)):
    x.append(mean_shape0[i][0])
    y.append(mean_shape0[i][1])

print x
'''

path='img/'
imgs=Image.loadImages(path)
Image.drawProNormalization(imgs, Persons)

