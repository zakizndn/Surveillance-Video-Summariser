# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 09:13:50 2022

@author: ahmadzakizainudin
"""

# load necessary libraries
from tkinter import *
import datetime
import tkinter as tk
from tkinter import filedialog
from tkVideoPlayer import TkinterVideo

root = tk.Tk()
root.title("Video Player")
root.geometry("800x700+290+10")

frame = tk.Frame(root)
frame.pack()

lower_frame = tk.Frame(root , bg = "#FFFFFF")
lower_frame.pack ( fill="both", side = BOTTOM)

# define functions
def update_duration(event):
    """ updates the duration after finding the duration """
    duration = vid_player.video_info()["duration"]
    end_time["text"] = str(datetime.timedelta(seconds=duration))
    progress_slider["to"] = duration

def update_scale(event):
    """ updates the scale value """
    progress_value.set(vid_player.current_duration())

def load_video():
    """ loads the video """
    file_path = filedialog.askopenfilename()

    if file_path:
        vid_player.load(file_path)

        progress_slider.config(to=0, from_=0)
        play_pause_btn["text"] = "Play"
        progress_value.set(0)

def seek(value):
    """ used to seek a specific timeframe """
    vid_player.seek(int(value))

def skip(value: int):
    """ skip seconds """
    vid_player.seek(int(progress_slider.get())+value)
    progress_value.set(progress_slider.get() + value)

def play_pause():
    """ pauses and plays """
    if vid_player.is_paused():
        vid_player.play()
        play_pause_btn["text"] = "Pause"

    else:
        vid_player.pause()
        play_pause_btn["text"] = "Play"

def video_ended(event):
    """ handle video ended """
    progress_slider.set(progress_slider["to"])
    play_pause_btn["text"] = "Play"
    progress_slider.set(0)

# browse button   
load_btn = tk.Button(root, text="Browse",bg="#FFFFFF", font=("calibri",
      12, "bold"), command=load_video)
load_btn.pack(ipadx=12, ipady=4, anchor=tk.NW)

vid_player = TkinterVideo(root, scaled=True)
vid_player.pack(expand=True, fill="both")

# backward button  
Buttonbackward = PhotoImage(file="backward.png")
back = tk.Button(lower_frame, image=Buttonbackward, bd=0, height = 50, width =50,
       command=lambda: skip(-10)).pack(side = LEFT) #rewind 5 seconds

# play button  
play_pause_btn = tk.Button(lower_frame, text="Play", width = 40, height = 2, command = play_pause)
play_pause_btn.pack(expand=True, fill="both", side = LEFT)

# forward button  
ButtonPlay = PhotoImage(file="forward.png")
Playbutton = tk.Button(lower_frame, image=ButtonPlay, bd=0,  height = 50, width =50,
       command=lambda: skip(10)).pack(side = LEFT) #fast forward 5 seconds

start_time = tk.Label(root, text=str(datetime.timedelta(seconds=0)))
start_time.pack(side="left")

progress_value = tk.IntVar(root)

progress_slider = tk.Scale(root, variable=progress_value, from_=0, to=0, orient="horizontal", command=seek)

progress_slider.pack(side="left", fill="x", expand=True)

end_time = tk.Label(root, text=str(datetime.timedelta(seconds=0)))
end_time.pack(side="left")

vid_player.bind("<<Duration>>", update_duration)
vid_player.bind("<<SecondChanged>>", update_scale)
vid_player.bind("<<Ended>>", video_ended )

root.mainloop()