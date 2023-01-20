#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 12:09:34 2022

@author: ahmadzakizainudin
"""

#%% Question 1 (Section 1) - by Ahmad Zaki 1191101837

import cv2
import os

working_path = '/Users/ahmadzakizainudin/Desktop/Assignment 1'
filename = 'surveillance_6.mp4'
filename = os.path.join(working_path, filename)
videoFile = cv2.VideoCapture(filename)


#%%

# video frame rate, resolution, and size
def getVideoProperty(videoFile):
    vFrameRate = int(videoFile.get(cv2.CAP_PROP_FPS))
 
    resolution = int(videoFile.get(cv2.CAP_PROP_FRAME_HEIGHT))
 
    videoSize = os.path.getsize(filename)
    
    return(vFrameRate, resolution, videoSize)

vFrameRate, resolution, videoSize = getVideoProperty(videoFile)


#%%

#extra
def getVideoProperty(videoFile):
    
    width = int(videoFile.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(videoFile.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    videoSize = os.path.getsize(filename)
    
    def convert_bytes(size):
        for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024.0:
                return "%3.1f %s" % (size, x)
            size /= 1024.0
            
    videoSize = os.path.getsize(filename)
    x = convert_bytes(videoSize)
    
    return(width, height, x)

width, height, x = getVideoProperty(videoFile)


#%%

#print result
print("Surveillance_6.mp4\n") 

print("Video frame rate: {} fps".format(vFrameRate)) # => 24 fps

print("Resolution: {} x {}".format(width, height)) # => 1280 x 720
print("Resolution: {}p".format(resolution)) # => 720p
 
print("Video size:", videoSize, 'bytes') # => 166,104,235 bytes
print("Video size:", x) # => 158.4 MB




