import numpy as np
import cv2
import keras
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import subprocess
from pycaw.pycaw import AudioUtilities, ISimpleAudioVolume, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
from ctypes import cast, POINTER
import ctypes
import wmi
import os



# Load the pre-trained model
model = keras.models.load_model(r"C:\Users\Danusri\MLPROJECT 2.0\best_model_dataflair3.h5")

# Constants for region of interest (ROI)
ROI_top = 100
ROI_bottom = 300
ROI_right = 150
ROI_left = 350

# Dictionary mapping gestures to actions
action_dict = {
    0: 'one',
    1: 'two',
    2: 'three',
    3: 'four',
    4: 'five',
    5: 'six',
    6: 'seven',
    7: 'eight',
    8: 'nine',
    9: 'ten'
}

def increase_volume():
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    current_volume = volume.GetMasterVolumeLevelScalar()
    new_volume = min(1.0, current_volume + 0.1)  
    volume.SetMasterVolumeLevelScalar(new_volume, None)

def decrease_volume():
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(ISimpleAudioVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(ISimpleAudioVolume))
    current_volume = volume.GetMasterVolume()
    new_volume = max(0.0, current_volume - 0.1)  
    volume.SyyyetMasterVolume(new_volume, None)

# Function to increase brightness
def increase_brightness():
    brightness_value = 50  
   # Connect to WMI
    wmi_obj = wmi.WMI(namespace='wmi')

    # Iterate through monitors
    monitors = wmi_obj.WmiMonitorBrightnessMethods()
    for monitor in monitors:
        monitor.WmiSetBrightness(brightness_value, 0)
        
def decrease_brightness():
    brightness_value = 30   

    # Connect to WMI
    wmi_obj = wmi.WMI(namespace='wmi')

    # Iterate through monitors
    monitors = wmi_obj.WmiMonitorBrightnessMethods()
    for monitor in monitors:
        monitor.WmiSetBrightness(brightness_value, 0)

accumulated_weight = 0.5


# Function to perform actions based on recognized gestures
def perform_action(action):
    if action == 'one':
        print("Performing action 1")
        print("Decreasing Brightness")
        decrease_brightness()
    
    elif action == 'two':
        print("Performing action 2")
        print("Opening Notepad")
        subprocess.Popen(['C:\\Users\\Danusri\\AppData\\Local\\Microsoft\\WindowsApps\\notepad.exe'])
    
    elif action == 'three':
        print("Performing action 3")
        print("Increasing Volume")
        increase_volume()
            
    elif action == 'four':
        print("Performing action 4")
        print("Opening Pictures")
        pictures_folder = os.path.expanduser('C:/Users/Danusri/OneDrive/Pictures')  # Path to the Pictures folder
        os.startfile(pictures_folder)
        
    elif action == 'five':
        print("Performing action 5")
        print("Increasing Brightness")
        increase_brightness()
    
    elif action == 'six':
        print("Performing action 6")
        print("Opening Pictures")
        pictures_folder = os.path.expanduser('C:/Users/Danusri/OneDrive/Pictures')  # Path to the Pictures folder
        os.startfile(pictures_folder)
        
    elif action == 'seven':
        print("Performing action 7")
        print("Opening Notepad")
        subprocess.Popen(['C:\\Users\\Danusri\\AppData\\Local\\Microsoft\\WindowsApps\\notepad.exe'])
        
    elif action == 'eight':
        print("Performing action 8")
        print("SOPTIFY IS OPENING")
        subprocess.Popen(['C:\\Users\\Danusri\\AppData\\Local\\Microsoft\\WindowsApps\\Spotify.exe'])
    
    elif action == 'nine':
        print("Performing action 9")
        print("Decreasing Volume")
        decrease_volume()
    
    elif action == 'ten':
        print("Performing action 10")
        print("Increasing Brightness")
        increase_brightness()

# Function to calculate accumulated average for background
def cal_accum_avg(frame, accumulated_weight):
    global background
    
    if background is None:
        background = frame.copy().astype("float")
        return None

    cv2.accumulateWeighted(frame, background, accumulated_weight)

# Function to segment the hand region
def segment_hand(frame, threshold=25):
    global background
    
    diff = cv2.absdiff(background.astype("uint8"), frame)
    _ , thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None
    else:
        hand_segment_max_cont = max(contours, key=cv2.contourArea)
        return thresholded, hand_segment_max_cont

# Video capture from the camera
cam = cv2.VideoCapture(0)
num_frames = 0
background = None

action_performed = False

while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)
    frame_copy = frame.copy()

    roi = frame[ROI_top:ROI_bottom, ROI_right:ROI_left]
    gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)

    if num_frames < 70:
        cal_accum_avg(gray_frame, accumulated_weight)
        if num_frames <= 59:
            cv2.putText(frame_copy, "FETCHING BACKGROUND...PLEASE WAIT", (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
    else:
        hand = segment_hand(gray_frame)

        if hand is not None:
            thresholded, hand_segment = hand
            cv2.drawContours(frame_copy, [hand_segment + (ROI_right, ROI_top)], -1, (255, 0, 0), 1)
            cv2.imshow("Thresholded Hand Image", thresholded)

            if num_frames > 300 and not action_performed:  # Check if action hasn't been performed yet
                thresholded_resized = cv2.resize(thresholded, (64, 64))
                thresholded_resized = cv2.cvtColor(thresholded_resized, cv2.COLOR_GRAY2RGB)
                thresholded_resized = np.reshape(thresholded_resized, (1, thresholded_resized.shape[0], thresholded_resized.shape[1], 3))

                pred = model.predict(thresholded_resized)
                gesture_index = np.argmax(pred)
                
                if gesture_index in action_dict:
                    action = action_dict[gesture_index]
                    print("Detected Gesture:", action)
                    perform_action(action)
                    action_performed = True  # Set flag to True after performing the action
       
    if action_performed:
        break


    cv2.rectangle(frame_copy, (ROI_left, ROI_top), (ROI_right, ROI_bottom), (255,128,0), 3)
    cv2.putText(frame_copy, "DataFlair hand sign recognition_ _ _", (10, 20), cv2.FONT_ITALIC, 0.5, (51,255,51), 1)
    cv2.imshow("Sign Detection", frame_copy)

    num_frames += 1
    k = cv2.waitKey(1) & 0xF
    if k == 27:
        break


# Release the camera and destroy all the windows
cam.release()
cv2.destroyAllWindows()
