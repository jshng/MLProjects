# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 09:19:31 2019

@author: Joseph.Shingleton
"""

from PIL import Image
import os
import os.path
import cv2
from Image_Preprocessing import *

image_path = 'original_images'

path, dirs, files = next(os.walk(image_path))
        
total_images = len(files) 
        
def get_image_array(image_file):
    image = cv2.imread(image_path + '\\' + image_file)
    
    return image      
        
def get_min_im_width(image_files):
    
    curr_min_width = []
    idx = 0
    for file in image_files:
        image = get_image_array(file)
        
        width, height, channels = np.shape(image)
        
        if idx ==0 or width < curr_min_width:
            curr_min_width = width
        else:
            continue

        idx += 1
    return curr_min_width


def get_min_im_height(image_files):
    
    curr_min_height = []
    idx = 0
    for file in image_files:
        image = get_image_array(file)
        
        width, height, channels = np.shape(image)
        
        if idx ==0 or height < curr_min_height:
            curr_min_height = height
        else:
            continue

        idx += 1
    return curr_min_height



def crop_and_rotate(image_files, save_dir, cropx_loc = 0, cropy_loc = 0):
     
    width_dim = 648
    height_dim = 1024
    idx = 0
    for file in files:
        
        img = get_image_array(file)
        rot_img1 = random_rotation(img, 25, channel_axis=2)
        rot_img2 = random_rotation(img, 25, channel_axis=2)
        
        crp_img = img[0:width_dim, 0:height_dim]
        crp_rot_img1 = rot_img1[0:width_dim, 0:height_dim]
        crp_rot_img2 = rot_img2[0:width_dim, 0:height_dim]
        
        
        cv2.imwrite(save_dir + 'ct' + str(idx) + '.PNG', crp_img)
        cv2.imwrite(save_dir + 'ct' + str(idx+len(files)) + '.PNG', crp_rot_img1)
        cv2.imwrite(save_dir + 'ct' + str(idx + 2*len(files)) + '.PNG', crp_rot_img2)
        
        
        idx += 1
        
        
        
        
    
    
    
    
    
    
    