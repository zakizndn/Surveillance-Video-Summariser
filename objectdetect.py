# -*- coding: utf-8 -*-

'''
Test out yolov5 code
Adapted from https://github.com/niconielsen32/ComputerVision/blob/master/DeployYOLOmodel.py
Video at https://www.youtube.com/watch?v=3wdqO_vYMpA
'''

#%%

import os
import cv2
import torch
import glob
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
        

#%% Read all saved frames in folder 'data' and detect human/items 

working_path = '/Users/ahmadzakizainudin/Desktop/Assignment 1'
working_path1 = "/Users/ahmadzakizainudin/Desktop/Assignment 1/person detected"
working_path2 = '/Users/ahmadzakizainudin/Desktop/Assignment 1/data'

# change directory    
os.chdir(working_path)

try:
	# creating a folder named person detected
	if not os.path.exists('person detected'):
		os.makedirs('person detected')

# if not created then raise error
except OSError:
	print ('Error: Creating directory of person detected')
    

#%% 

for filename in os.listdir(working_path2):
    if (filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg")):
        print(filename)
        
        
#%% 

for filename in natsort.natsorted(os.listdir(working_path2),reverse = False):
    if (filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg")):
        print(filename)

        
#%% 
    
obj = ObjectDetection()

# sort filename in the correct order
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
    
    
#%% 

folder_dir = "/Users/ahmadzakizainudin/Desktop/Assignment 1/media"
folder_dir1 = "/Users/ahmadzakizainudin/Desktop/Assignment 1/person detected"

# read a frame
obj = ObjectDetection()

for filename in os.listdir(folder_dir):
    # check if the image ends with png or jpg or jpeg
    if (filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg")):
        x = 0
        imgfile = os.path.join(folder_dir, filename) 
        img = cv2.imread(imgfile)  
       
        # run detection on the image
        img_out, results_out = obj.runDetection(img)
        
        labels, cords = results_out
        print('reading ' + filename)
        n = len(labels)
        for i in range(n):
            # print all detected objects in a frame
            print(int(labels[i]), ' label :' ,  obj.class_to_label(labels[i])) 
            if int(labels[i]) == 0:
                x += 1
    
        # if human is detected, save current frame in different folder
        if x == 1:
            os.chdir(folder_dir1)
            print(f'{x} person is detected')
            print('Creating...' + filename)
            cv2.imwrite(filename, img)
             
        # >1 person are detected
        elif x > 1:
            os.chdir(folder_dir1)
            print(f'{x} people are detected')
            print('Creating...' + filename)
            cv2.imwrite(filename, img)
           
        # no human is detected
        elif int(labels[i]) != 0:
            print('No person is detected. skipped')
        
        print('\n')
    

#%% test 1 - list all detected objects in all frames

folder_dir = "/Users/ahmadzakizainudin/Desktop/Assignment 1/data"
obj = ObjectDetection()

for filename in os.listdir(folder_dir):
    # check if the image ends with png or jpg or jpeg
    if (filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg")):
        imgfile = os.path.join(folder_dir, filename) 
        img = cv2.imread(imgfile)  
        img_out, results_out = obj.runDetection(img)
        
        labels, cords = results_out
        print(filename)
        n = len(labels)
        for i in range(n):
           print(int(labels[i]), ' label :' ,  obj.class_to_label(labels[i]))
        print('\n')
    
    
#%% test 2 - create folder person detected

working_path = '/Users/ahmadzakizainudin/Desktop/Assignment 1'
os.chdir(working_path)

try:
	# creating a folder named data
	if not os.path.exists('person detected'):
		os.makedirs('person detected')

# if not created then raise error
except OSError:
	print ('Error: Creating directory of person detected')


#%% test 2 - save all frames containing human in separate folder

folder_dir = "/Users/ahmadzakizainudin/Desktop/Assignment 1/media"
folder_dir1 = "/Users/ahmadzakizainudin/Desktop/Assignment 1/person detected"

# read a frame
obj = ObjectDetection()

for filename in os.listdir(folder_dir):
    # check if the image ends with png or jpg or jpeg
    if (filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg")):
        x = 0
        imgfile = os.path.join(folder_dir, filename) 
        img = cv2.imread(imgfile)  
       
        # run detection on the image
        img_out, results_out = obj.runDetection(img)
        
        labels, cords = results_out
        print('reading ' + filename)
        n = len(labels)
        for i in range(n):
            # print all detected objects in a frame
            print(int(labels[i]), ' label :' ,  obj.class_to_label(labels[i])) 
            if int(labels[i]) == 0:
                x += 1
    
        # if human is detected, save current frame in different folder
        if x == 1:
            os.chdir(folder_dir1)
            print(f'{x} person is detected')
            print('Creating...' + filename)
            cv2.imwrite(filename, img)
             
        # >1 person are detected
        elif x > 1:
            os.chdir(folder_dir1)
            print(f'{x} people are detected')
            print('Creating...' + filename)
            cv2.imwrite(filename, img)
           
        # no human is detected
        elif int(labels[i]) != 0:
            print('No person is detected. skipped')
        
        print('\n')
        

#%% test 3 - read video, read frame one by one, only save those with human in it xxxxx

folder_dir = "/Users/ahmadzakizainudin/Desktop/Assignment 1/media"
folder_dir1 = "/Users/ahmadzakizainudin/Desktop/Assignment 1/person detected"
folder_dir2 = "/Users/ahmadzakizainudin/Desktop/Assignment 1/data"

working_path = '/Users/ahmadzakizainudin/Desktop/Assignment 1'
#filename = 'surveillance_6.mp4'
filename = 'cctv62.mp4'
filename = os.path.join(working_path,filename)
videoFile = cv2.VideoCapture(filename)

# read a frame
obj = ObjectDetection()

duration = 1

vFrameRate = int(videoFile.get(cv2.CAP_PROP_FPS))
duration = duration * 60 * vFrameRate
timer = 0
framelist = 0
     
while timer < duration:
     # reading from frame
     ret, frame = videoFile.read()
     timer += 1
     
     if ret:
         x = 0
         img = cv2.imread(frame)  
         img_out, results_out = obj.runDetection(img)
         
         filename = 'frame' + str(framelist) + '.jpg'
         
         labels, cords = results_out
         print('reading ' + filename)
         n = len(labels)
         for i in range(n): 
             if int(labels[i]) == 0:
                 x += 1
         # if human is detected, save current frame in different folder
         if x == 1:
             os.chdir(folder_dir2)
             print(f'{x} person is detected')
             print('Creating...' + filename)
             cv2.imwrite(filename, img)
              
         # >1 person are detected
         elif x > 1:
             os.chdir(folder_dir2)
             print(f'{x} people are detected')
             print('Creating...' + filename)
             cv2.imwrite(filename, img)
            
         # no human is detected
         elif int(labels[i]) != 0:
             print('No person is detected. skipped')
         
         print('\n')
                 
         
         # if video is still left continue creating images
         filename = 'data/frame' + str(framelist) + '.jpg'
         print ('Creating...' + filename)
         
         # increasing counter so that it will show how many frames are created
         framelist += 1
         
     else:
         break
             
# Release all space and windows once done
videoFile.release()
cv2.destroyAllWindows()


#%% assgin frames to an array (framelist)

path = "/Users/ahmadzakizainudin/Desktop/Assignment 1/data"
framesList = os.listdir(path)      
print (framesList)

for i,imf in enumerate(framesList[:20]):
    print(i, imf)
    imf = os.path.join(path, imf) 
    print(imf)
    
    
#%%

path = "/Users/ahmadzakizainudin/Desktop/Assignment 1/data"
fig = plt.figure()

# Visualize the first 20 frames
for i, imf in enumerate(framesList[:20]):
    fig.add_subplot(4, 5, i+1) # display image in 4x5 grid
    imf = os.path.join(path, imf) 
    imf_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im = cv2.imread(imf_rgb)
    plt.imshow(im)
    plt.axis('off')


#%%

path = "/Users/ahmadzakizainudin/Desktop/Assignment 1/data"
y = 0

fig = plt.figure()

for filename in os.listdir(path):
    # check if the image ends with png or jpg or jpeg
    if (filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg")) and y <= 20:
        imgfile = os.path.join(path, filename) 
        img = cv2.imread(imgfile)  
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.axis('off')
        
    elif y > 20:
        break
    
    y += 1
    
    
#%% Visualize detection output

# opencv read image into BGR format , convert to RGB
plt.figure(figsize = (10,5))
img_rgb = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb) , plt.title('Image Output After Detection by YoloV5') 
plt.axis('off')


#%% Check object detected in the image

labels, cords = results_out
class_list = obj.get_class_list()
n = len(labels)
for i in range(n):
   print( int( labels[i]))
   
print('Objects detected in the target image')
for i in range(n):
   print( int( labels[i]) ,  ' label :' ,  obj.class_to_label(labels[i])  )
    
   
    

