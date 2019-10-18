# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 15:13:21 2019

- Downloads Cifar10 image set with batch id 'batch_id'
- Splits into training and test images
- Applies affine transform to training images

@author: Joseph.Shingleton
"""

import numpy as np
import pickle
from Image_Preprocessing import *


def load_cifar10_batch(batch_id, CIFAR10_DATASET_FOLDER):
    """Loads Cifar10 images for model training
    arguments:
        batch_id = batch number to be downloaded (int)
        CIFAR10_DATASET_FOLDER = where are you getting the data from?
    returns:
        Images (3d array) and their corresponding labels (list of ints)    
    """
    with open(CIFAR10_DATASET_FOLDER + '/data_batch_' + str(batch_id), 
              mode = 'rb') as file:
        batch = pickle.load(file, encoding = 'latin1')

    features = batch['data'].reshape((len(batch['data']),3,32,32)).transpose(0,2,3,1)
    
    labels = batch['labels']
    
    return features, labels


def train_test_split(features, labels, ratio):
    """Splits a data set into training and test data
    arguments: 
        features = images from dataset as 3d arrays
        labels   = corresponding labbels as list of integers
        ratio    = positive float <1, e.g. 0.6 == 60:40
    returns:
        Original data split into testing and training 
    **NOTE: If augmenting data it may be necessary to use a lower ratio**
    """
    train_size = int(len(features)*ratio)
    
    training_images = features[:train_size,:,:]
    training_labels = labels[:train_size]
    
    test_images = features[train_size:]
    test_labels = labels[train_size:]

    return training_images, training_labels, test_images, test_labels


def augment_data(imgs, labels, augs_per_image, theta_range=0, tx_range=0, 
                      ty_range=0, shear_range=0, zx_range = 1, zy_range = 1,
                      bright_range = None):
    """Applies a number of random affine transformations to each image in a 
    data set. 
    # Arguments
        imgs           = list images converted to 2d numpy arrays
        labels         = list of image lables
        augs_per_image = number of times to aply random transform (int)
        theta_range    = rotation range in degrees (float, float)
        tx_range       = horizontal shift proportion range (float, float)
        ty_range       = vertical shift proportion range (float, float)
        shear_range    = shear angle range in degrees (float, float)
        zx_range       = horzontal zoom range (float, float)
        zy_range       = verticle zoom range (float, float)
        **Note: leave argument blank if no particular transformation is 
        not required
    # Returns
        list of np arrays of length augs_per_image*len(imgs) representing 
        transormed images, similar list of labels.
    """
    # Create storage vectors
    augmented_images = []
    augmented_labels = []
    
    for ii in range(len(imgs)):
        
        for jj in range(augs_per_image):
            manipulated_img = random_affine_transform(imgs[ii], theta_range,
                                                      tx_range, ty_range,
                                                      shear_range, zx_range, 
                                                      zy_range)
            if bright_range != None:
                manipulated_img = random_brightness(manipulated_img, 
                                                    bright_range)
            # Add manipulated images to augmented_images and label
            augmented_images.append(manipulated_img)
            augmented_labels.append(labels[ii])
        
    return augmented_images, augmented_labels


def shuffle_paired_list(A,B):
    """Allows for a list to be shuffled while maintaining coupled locations
    Arguments:
        A,B = lists of equal length. Shapes need not be equal
    Returns:
        Shuffled lists with maintained coupling
    """
    combined = list(zip(A, B))
    np.random.shuffle(combined)
    A[:], B[:] = zip(*combined)  
    
    return A, B  

