import av
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import traceback
from gcsfs import GCSFileSystem
from streamlit_webrtc import (webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration)
from shared_functions import (mediapipe_detection, extract_key_points, display_gif, display_gesture_checkboxes)

# Initialize a Mediapipe Holistic object
mp_holistic = mp.solutions.holistic

# Initialize a GCS file system object
fs = GCSFileSystem(project='keras-file-storage')

# Specify the path to the model file in the GCS bucket
model_path = 'gs://keras-files/lesson4.keras'
local_model_path = 'lesson4.keras'

# Download the model file from GCS to local file system
try:
    with fs.open(model_path, 'rb') as f_in:
        with open(local_model_path, 'wb') as f_out:
            f_out.write(f_in.read())

except Exception as e:
    # Display error message if model download fails
    st.error(f"Error downloading the model: {e}")
    st.error(f"Exception traceback: {traceback.format_exc()}")
    st.stop()

# Load the model outside the function
try:
    # Load the trained model from the local file system
    lesson4_model = tf.keras.models.load_model(local_model_path, compile=False)
    st.write("Model loaded successfully.")  # Debug statement
except Exception as e:
    # Display error message if model loading fails
    st.error(f"Error loading the model: {e}")
    st.error(f"Exception traceback: {traceback.format_exc()}")
    st.stop()


# Define a custom video processor class inheriting from VideoProcessorBase
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = lesson4_model  # Initialize model attribute with loaded model
        self.actions = np.array(['good', 'happy', 'hearing', 'mine', 'no', 'yes', 'what', 'where', 'who', 'you',
                                 'yours'])
        self.sequence = []  # Initialize an empty list to store key point sequences
        self.sentence = []  # Initialize an empty list to store recognized sentences
        self.threshold = 0.4  # Define a confidence threshold
        self.mp_holistic = mp.solutions.holistic  # Initialize Mediapipe Holistic object

    # Define a method to process video frames
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")  # Convert video frame to NumPy array

        # Mediapipe processing
        with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            image, results = mediapipe_detection(img, holistic)  # Perform detection using Mediapipe
            key_points = extract_key_points(results)  # Extract key points from detection results
            self.sequence.append(key_points)  # Append key points to sequence list
            self.sequence = self.sequence[-30:]  # Keep only the last 30 frames in the sequence

            # Perform prediction when sequence length reaches 30
            if len(self.sequence) == 30:
                res = self.model.predict(np.expand_dims(self.sequence, axis=0))[0]  # Perform model prediction
                predicted_action_index = np.argmax(res)  # Get index of predicted action
                if res[predicted_action_index] > self.threshold:  # Check if prediction confidence is above threshold
                    self.sentence.append(
                        self.actions[predicted_action_index])  # Append predicted action to sentence list

            if len(self.sentence) > 5:  # Keep only the last 5 sentences
                self.sentence = self.sentence[-5:]

            if len(self.sentence) > 0:  # Add recognized sentence to the video frame
                cv2.putText(image, self.sentence[-1], (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        # Convert the modified image back to a video frame and return it
        return av.VideoFrame.from_ndarray(image, format="bgr24")


# Define RTC configuration for WebRTC
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})


def lesson_page_4():
    st.title("Lesson 4")
    st.write("Select any of the gestures you'd like to see. Deselect them if you no longer need them. When you are "
             "ready, select 'Start Camera' to begin practicing the gestures. Remember to go slow and try a few times.")
    st.write(" In this lesson, we will practice the following gestures:")

    # Define a dictionary mapping gesture names to GIF paths
    gesture_gifs = {
        "good": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExMW1kamFna25scG4yY2lvandqc3Jjd2dtcmxjNW5hdTRsc3VoeXRrYSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/tizTuXYdDa2MTks9Tc/giphy.gif",
        "happy": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExbXJ2cHg2ZDJ2dW9rcHJkN2kzNHR4Ymdscm95b21jcDkxanE0MXhseSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/MQNTBwdaLgODdUQaNz/giphy.gif",
        "hearing": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExeTR0OGpycjQxZ2pvbDRiZHVjN2VqYnpnNTNlZ3cwNjBtdXJ5M3dpbSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/99sSdjMTz0e8epZcGF/giphy.gif",
        "mine": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExd3lzajFtMDZydmp3bGc5M2s1NnJkbmFqd2M3a3RjZ2R1NTEyZmluNyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/BsM0mfAxTEV669h1BH/giphy.gif",
        "no": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExbHVpZmMxeXcwbHEzODliM3hkMDRyODVmOHhvbnR2a2RueXdoeGh6cyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/0DlkKenmPE204WNJyK/giphy.gif",
        "yes": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExZDhxcnY5eXVrMnJnOHU2N2V3b3N0cGhzdmVrYXE0dHdlamZ6bWo2YSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/2RAmynm5D7X5Ls1MSQ/giphy.gif",
        "what": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExYXo4bDBicHByNTFjaW9ucWM1bWI2cjJtM3ZodjZ2ZWY2MGt3ZGNqeiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/n1KQ4g5MhPzzkeDDKQ/giphy.gif",
        "where": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExODNyMmZ1NGgzcW1kc3p2cTJwajJsbXA2dnB0djdzcG42MWxkenMxZCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/mb0Ts30lv53qsytio9/giphy.gif",
        "who": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExNW53d2c0bTlic2FyOGh1bmNhamtmeDh5aDQ0cmZnbjQzNDYzbGE5MCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/LqKZxO4UHF3JcD1W9S/giphy.gif",
        "you": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExZDJ5cGZleWVmOG1nbmIyNWp6Nms3ZXI2NW9rZWk3cG1tM3czcG1heSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/YWhNR9Z055inTOcBaL/giphy.gif",
        "yours": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExZ2lpMDc4OHpsZDJudm40b20wbml6Nmw3amtkdWxoNWhhYjlydGhraSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/6VaSpEhIcQuLuADHwO/giphy.gif"
    }

    # Display gesture checkboxes and get selected gestures
    selected_gestures = display_gesture_checkboxes(gesture_gifs)

    # Iterate over selected gestures and display corresponding GIFs
    for gesture_name, selected in selected_gestures.items():
        if selected:
            display_gif(gif_path=gesture_gifs[gesture_name], gesture_name=gesture_name)

    # Start WebRTC streaming session
    webrtc_streamer(
        key="example",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        async_processing=True,
    )
