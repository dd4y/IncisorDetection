'''
Created on 06.05.2014.

@author: d.day
'''
import numpy as np

class Incisor:

    def __init__(self, persID,incID):
        self.person=persID
        self.ID = incID

        filename='lm/landmarks'+str(persID)+'-'+str(incID)+'.txt'
        f = open(filename, 'r')
        Lf=[]
        for line in f:
            Lf=Lf+[int(float(line))]

        self.xy=np.empty((len(Lf)/2,2),np.int32)
        self.normXY=np.empty((len(Lf)/2,2),np.int32)
        for x in range(len(Lf)/2):
            self.xy[x]=Lf[2*x],Lf[2*x+1]

        self.centroid=np.int32(self.xy.mean(0))


    def displayIncesor(self):
        print "Person : ", self.person,  ", Incisor: ", self.ID, ", Landmarks: ", self.xy
