# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 09:19:31 2019

@author: Joseph.Shingleton
"""

import PIL
import os
import os.path

image_path = '/originalimages/'

path, dirs, files = next(os.walk(image_path))

total_images = len(files) 