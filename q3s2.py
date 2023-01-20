#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 22:34:34 2022

@author: ahmadzakizainudin
"""

#%% Question 3 (Section 2) - by Ahmad Zaki 1191101837

import os
import cv2
import natsort
from PIL import Image


#%%

# define working path
working_path = '/Users/ahmadzakizainudin/Desktop/Assignment 1'
working_path1 = '/Users/ahmadzakizainudin/Desktop/Assignment 1/data'

# create folder compressed image
os.chdir(working_path)
try:
	if not os.path.exists('compressed image'):
		os.makedirs('compressed image')

# if not created then raise error
except OSError:
	print ('Error: Creating directory of compressed image')
    
    
#%%

# check frames order 
for framelist in os.listdir(working_path1):
    print(framelist)        
    
# frames are not in the correct order 


#%%

os.chdir(working_path1)

# sort frames in the correct order, store them in a list
filepath = []

for framelist in natsort.natsorted(os.listdir(working_path1),reverse = False):
    if (framelist.endswith(".png") or framelist.endswith(".jpg") or framelist.endswith(".jpeg")):
        framelist = os.path.join(os.getcwd(), framelist)
        filepath.append(framelist)
        
#print(filepath)        


# sort frames in the correct order, store them in a list
filepath1 = []

for framelist in natsort.natsorted(os.listdir(working_path1),reverse = False):
    if (framelist.endswith(".png") or framelist.endswith(".jpg") or framelist.endswith(".jpeg")):
        filepath1.append(framelist)
        
#print(filepath1)  


#%%     

# define working path
working_path2 = '/Users/ahmadzakizainudin/Desktop/Assignment 1/compressed image'

for i, imf in enumerate(filepath):    
    os.chdir(working_path2)
    picture = Image.open(imf)
    print('compressing', filepath1[i])
    picture.save("compressed_"+filepath1[i], "JPEG", optimize = True, quality = 10)


#%% 

# sort frames in the correct order, store them in a list
filepath = []

for framelist in natsort.natsorted(os.listdir(working_path2),reverse = False):
    if (framelist.endswith(".png") or framelist.endswith(".jpg") or framelist.endswith(".jpeg")):
        framelist = os.path.join(os.getcwd(), framelist)
        filepath.append(framelist)
      
        
#%%

img_array = []
for file in filepath:
    img = cv2.imread(file)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)
    

#%% Motion JPEG (MJPEG) 

os.chdir(working_path)
out = cv2.VideoWriter('compressed_video.avi', cv2.VideoWriter_fourcc(*'DIVX'), 10, size) #10 frames per second
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()


#%%

filename = 'surveillance_6.mp4'
filename = os.path.join(working_path,filename)
videoFile = cv2.VideoCapture(filename)


#%%

def encodeVideoAsMJPEG(videoFile, duration):
    destinationFolder = '/Users/ahmadzakizainudin/Desktop/Assignment 1'
    os.chdir(destinationFolder)
    out = cv2.VideoWriter('compressed_video.avi', cv2.VideoWriter_fourcc(*'DIVX'), 10, size)
     
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    
    os.chdir(working_path1)
    size1 = os.path.getsize('frame0.jpg')
    
    os.chdir(working_path2)
    size2 = os.path.getsize('compressed_frame0.jpg')
    
    compressRatio = size1 / size2
    
    return (destinationFolder, compressRatio)

folder, CR = encodeVideoAsMJPEG(videoFile, 10) #10 minutes 

print('Compression Ratio is '+ str((round(CR, 2)))+ ':1')




