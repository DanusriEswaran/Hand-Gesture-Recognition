import numpy as np
import cv2
import keras
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import subprocess

# Load the pre-trained model
model = keras.models.load_model(r"C:..\MLPROJECT\best_model_dataflair3.h5")

# Constants for region of interest (ROI)
ROI_top = 100
ROI_bottom = 300
ROI_right = 150
ROI_left = 350

# Dictionary mapping gestures to actions
action_dict = {
    0: 'open',
    1: 'play',
    2: 'volume_increase',
    3: 'volume_decrease',
    4: 'mute',
    5: 'next',
    6: 'previous',
    7: 'add_to_queue',
    8: 'stop',
    9: 'close'
}

accumulated_weight = 0.5


# Function to perform actions based on recognized gestures
def perform_action(action):
    if action == 'open':
        print("Performing 'open' action...")
        subprocess.Popen(['..\\Spotify.exe'])
    elif action == 'play':
        print("Performing 'play' action...")
    elif action == 'volume_increase':
        print("Performing 'volume increase' action...")
    elif action == 'volume_decrease':
        print("Performing 'volume decrease' action...")
    elif action == 'mute':
        print("Performing 'mute' action...")
    elif action == 'next':
        print("Performing 'next' action...")
    elif action == 'previous':
        print("Performing 'previous' action...")
    elif action == 'add_to_queue':
        print("Performing 'add to queue / favourites' action...")
    elif action == 'stop':
        print("Performing 'stop' action...")
    elif action == 'close':
        print("Performing 'close' action...")

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

            if num_frames > 300:
                thresholded_resized = cv2.resize(thresholded, (64, 64))
                thresholded_resized = cv2.cvtColor(thresholded_resized, cv2.COLOR_GRAY2RGB)
                thresholded_resized = np.reshape(thresholded_resized, (1, thresholded_resized.shape[0], thresholded_resized.shape[1], 3))

                pred = model.predict(thresholded_resized)
                gesture_index = np.argmax(pred)
                
                if gesture_index in action_dict:
                    action = action_dict[gesture_index]
                    print("Detected Gesture:", action)
                    perform_action(action)


    cv2.rectangle(frame_copy, (ROI_left, ROI_top), (ROI_right, ROI_bottom), (255,128,0), 3)
    cv2.putText(frame_copy, "DataFlair hand sign recognition_ _ _", (10, 20), cv2.FONT_ITALIC, 0.5, (51,255,51), 1)
    cv2.imshow("Sign Detection", frame_copy)

    num_frames += 1
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break


# Release the camera and destroy all the windows
cam.release()
cv2.destroyAllWindows()
