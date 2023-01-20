#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
Created on Thu Dec  1 20:36:59 2022

@author: ahmadzakizainudin
"""

#%% Question 2 (Section 1) - by Ahmad Zaki 1191101837

import os
import cv2
import torch
import natsort 
from matplotlib import pyplot as plt


#%% Class for ObjectDetection

class ObjectDetection:
    """
    Class implements Yolo5 model to make inferences on a youtube video using OpenCV.
    """
    
    def __init__(self):
        """
        Initializes the class with youtube url and output file.
        :param url: Has to be as youtube URL,on which prediction is made.
        :param out_file: A valid output file name.
        """
        #self._URL = url
        self.model = self.load_model()
        self.classes = self.model.names
        #self.out_file = out_file
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("\n\nDevice Used:",self.device)

    def load_model(self):
        """
        Loads Yolo5 model from pytorch hub.
        :return: Trained Pytorch model.
        """
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model

    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
     
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]

    def get_class_list(self):
        return self.classes


    def plot_boxes(self, results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

        return frame

    def runDetection(self, frame):
      results = self.score_frame(frame)
      frame = self.plot_boxes(results, frame)
      return frame, results


#%%

# define working path
working_path = '/Users/ahmadzakizainudin/Desktop/Assignment 1'
working_path1 = "/Users/ahmadzakizainudin/Desktop/Assignment 1/person detected"
working_path2 = '/Users/ahmadzakizainudin/Desktop/Assignment 1/data'

# get video file
filename = 'surveillance_6.mp4'
filename = os.path.join(working_path,filename)
videoFile = cv2.VideoCapture(filename)

# create folder data
os.chdir(working_path)
try:
	# creating a folder named data
	if not os.path.exists('data'):
		os.makedirs('data')

# if not created then raise error
except OSError:
	print ('Error: Creating directory of data')

# create folder person detected
os.chdir(working_path)
try:
	# creating a folder named person detected
	if not os.path.exists('person detected'):
		os.makedirs('person detected')

# if not created then raise error
except OSError:
	print ('Error: Creating directory of person detected')
  
    
#%% 

obj = ObjectDetection()

def summarizeVideo(videoFile, duration):
    
    vFrameRate = int(videoFile.get(cv2.CAP_PROP_FPS))
    # captering frame from a video every 1 seconds @ every 24 frames
    save_interval = 1
    frame_count = 0
    framelist1 = 0
    duration = duration * 60 #in seconds
    
    while videoFile.isOpened():
        ret, frame = videoFile.read()
        
        if ret:
            frame_count += 1

            if frame_count % (vFrameRate * save_interval) == 0:
                
                name = 'data/frame' + str(framelist1) + '.jpg'
                print ('Creating...' + name)
                
                # write frames in folder 'data'
                cv2.imwrite(name, frame)
              
                # optional 
                frame_count = 0
                framelist1 += 1
            
            # break the loop
            elif (framelist1 == duration/save_interval):
                break
            
        # break the loop
        else:
            break

    videoFile.release()
    cv2.destroyAllWindows()
    
    for filename in natsort.natsorted(os.listdir(working_path2),reverse = False):
        if (filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg")):
            x = 0
            imgfile = os.path.join(working_path2, filename) 
            img = cv2.imread(imgfile)  
            # run detection on the image
            img_out, results_out = obj.runDetection(img)
        
            labels, cords = results_out
            # read frame
            print('reading ' + filename)
            n = len(labels)
            for i in range(n):
                print(int(labels[i]), ' label :' ,  obj.class_to_label(labels[i]))
                if int(labels[i]) == 0:
                    x += 1
               
            # if human is detected, save current frame in different folder
            if x == 1:
                os.chdir(working_path1)
                print(f'{x} person is detected')
                print('Creating...' + filename)
                cv2.imwrite(filename, img)
             
            # >1 person are detected
            elif x > 1:
                os.chdir(working_path1)
                print(f'{x} people are detected')
                print('Creating...' + filename)
                cv2.imwrite(filename, img)
           
            # no human is detected
            elif x == 0:
                print('No person is detected. skipped')
        
            print('\n')
    
    return framelist1

framesList1 = summarizeVideo(videoFile, 10) #10 minutes 


#%%

# sort frames in the correct order, store them in a list
framesList = []

for framelist in natsort.natsorted(os.listdir(working_path1),reverse = False):
    if (framelist.endswith(".png") or framelist.endswith(".jpg") or framelist.endswith(".jpeg")):
        framelist = os.path.join(working_path1, framelist)
        # adding items to the list
        framesList.append(framelist)
          
#print(framesList)


#%%

# sort frames in the correct order, store them in a list
framesList1 = []
for framelist in natsort.natsorted(os.listdir(working_path1),reverse = False):
    if (framelist.endswith(".png") or framelist.endswith(".jpg") or framelist.endswith(".jpeg")):
        framesList1.append(framelist)

print(framesList1[:20])


#%% 

fig = plt.figure()
# change window title
fig.canvas.manager.set_window_title('Visualize the first 20 frames')

# Visualize the first 20 frames
for i, imf in enumerate(framesList[:20]):
    print("Visualizing frame " + imf)
    ax = fig.add_subplot(4, 5, i+1) # display image in 4x5 grid
    img = cv2.imread(imf)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb), ax.title.set_text(framesList1[i+1])
    plt.axis('off')




