import av
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from gcsfs import GCSFileSystem
from streamlit_webrtc import (webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration)
from shared_functions import (mediapipe_detection, extract_key_points, display_gif, display_gesture_checkboxes,
                              download_and_load_model)

# Initialize a Mediapipe Holistic object
mp_holistic = mp.solutions.holistic

# Initialize a GCS file system object
fs = GCSFileSystem(project='keras-file-storage')

# Specify the path to the model file in the GCS bucket
model_path = 'gs://keras-files/lesson3.keras'
local_model_path = 'lesson3.keras'

# Call function to download the model
lesson3_model = download_and_load_model(model_path, local_model_path)


class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = lesson3_model  # Initialize model attribute with loaded model
        self.actions = np.array(['name', 'need', 'now', 'please', 'sad', 'school', 'slow', 'sorry', 'wait', 'want'])
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


# Define a custom video processor class inheriting from VideoProcessorBase
def lesson_page_3():
    st.title("Lesson 3")
    st.write("Select any of the gestures you'd like to see. Deselect them if you no longer need them. When you are "
             "ready, select 'Start Camera' to begin practicing the gestures. Remember to go slow and try a few times.")
    st.write(" In this lesson, we will practice the following gestures:")

    # Define a dictionary mapping gesture names to GIF paths
    gesture_gifs = {
        "name": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExb3M0dGtmNXBlcDl4OGo3azFtcnp5cTdmeG1zMmNwN2wwc25ma2oyOCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/q0hxfOXZMLjdHm2GZv/giphy.gif",
        "need": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExb2NnMzhsajQ4bjd3b3BwNW16NnN2cjUyc2Z4ZjkwY2Zlam0wY3R6MyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/21N5uZTr8Zizwb0qGI/giphy.gif",
        "now": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExZHZsdjJmOTJ3a2ltbGd6a3RmcWxsYmlvMjV1eTA3YWVyY2E2c2I5NCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/shfpKTK1fwLsX3QLfP/giphy.gif",
        "please": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExdm5zOWRuNXB6MmkzcmFtZnN5YmtmM2k1Z2Z4bm9saXJxZjE1MGZ1eCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/jjQAlZCijQWWWKbAtF/giphy.gif",
        "sad": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExN2dtM3VqbnVpbG4wN3QzeXh0a2FwaG9rcmJuZTI1czdhaWZ0aTlkdiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/XHi2ODj1G0KkBGvevS/giphy.gif",
        "school": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExeGY2aXUwaWdoNmhia21pMTk0d282MXg3MWk4NDd4ZHd6aDlpemVoYiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/IIvfZVBqy00sKIHceT/giphy.gif",
        "slow": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExanQ5cm44N2NnZWl1NXZiaTRmaXF0bW9yMGVieG9kN3N3bXQ5N3RmMSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3uusRLnvmzoEwLwlwy/giphy.gif",
        "sorry": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExa2pkOG5pbGFlN3FtajJjaXNxejkwOGVhNmZ3dGVtY2Frc2E0OGEwayZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/5sSYFM8Zx1pvXft3G0/giphy.gif",
        "wait": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExYm53dmx1azFnNm5rYzJlNTJqOWR6bTE0aTZjN2lnaGw1Z2VyZ3lmdyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/YnP0kJnn1gScEYZ0WM/giphy.gif",
        "want": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExNDZ2bHQ2d3BjdDhwaGU4eDhrcWtya3hjYzdteW41Z2ZiNzNteWZycCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/u539NsiA5aefP6pYkd/giphy.gif"
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