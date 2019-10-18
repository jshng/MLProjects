# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 10:20:14 2019

@author: Joseph.Shingleton
"""
import numpy as np
import pickle
from augment_data import load_cifar10_batch
from matplotlib import pyplot as plt


def seperate_cifar10_classes():
    """ Separates all cifar10 images into classes
    arguments:
        None
    returns:
        10 arrays of categorised image data
    """
    airplane   = []
    automobile = []
    bird       = []
    cat        = []
    deer       = []
    dog        = []
    frog       = []
    horse      = []
    ship       = []
    truck      = []
    
    index = 0 
    
    for ii in range(1,6):
        data = load_cifar10_batch(ii, 'cifar-10-batches-py')               
       
        while index < 10000:
            
            if data[1][index] == 0:
                airplane.append(data[0][index])        
            elif data[1][index] == 1:
                automobile.append(data[0][index])
            elif data[1][index] == 2:
                bird.append(data[0][index])
            elif data[1][index] == 3:
                cat.append(data[0][index])
            elif data[1][index] == 4:
                deer.append(data[0][index])
            elif data[1][index] == 5:
                dog.append(data[0][index])
            elif data[1][index] == 6:
                frog.append(data[0][index])
            elif data[1][index] == 7:
                horse.append(data[0][index])
            elif data[1][index] == 8:
                ship.append(data[0][index])
            elif data[1][index] == 9:
                truck.append(data[0][index])
    
            index += 1
    # add labels to each class
    return airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
   
         
def recombine_cifar10_classes(*classes):
    
    N = len(classes)
    classes     = list(classes)
    classlabels = []
    images      = []
    labels      = []
   
    # Add labels to classes
    for ii in range(N):
        class_length = len(classes[ii])
        label        = np.ones(class_length)*ii
        classlabels.append((classes[ii], label))

    # Concatenate classes into single list
    for jj in range(N):
        images = images + classlabels[jj][0]
        labels = labels + list(classlabels[jj][1])
    
    labels = [int(i) for i in labels]
   
    # Shuffle images while maintaining labels
    zipped = list(zip(images, labels))
    np.random.shuffle(zipped)
    images[:], labels[:] = zip(*zipped)
    
    images = np.asarray(images)
    return images, labels
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
           
