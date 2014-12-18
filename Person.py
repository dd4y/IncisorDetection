'''
Created on 06.05.2014.

@author: d.day
'''
import Incisor
class Person:

    persCount=0;

    def __init__(self, persID):
        self.ID = persID
        self.Incisors=[]
        for i in range(1,9):
            self.Incisors.append(Incisor.Incisor(self.ID,i))

        Person.persCount += 1

    def displayCount(self):
        print "Total Persons %d" % Person.empCount

    def displayPerson(self):
        print "Name : ", self.ID,  ", Incisors number: ", len(self.Incisors)
