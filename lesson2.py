import traceback
import av
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from gcsfs import GCSFileSystem
from streamlit_webrtc import (webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration)
from shared_functions import (mediapipe_detection, extract_key_points, display_gif, display_gesture_checkboxes)

# Initialize a GCS file system object
fs = GCSFileSystem(project='keras-file-storage')

# Specify the path to the model file in the GCS bucket
model_path = 'gs://keras-files/lesson2.keras'
local_model_path = 'lesson2.keras'

# Load the model outside the function
try:
    # Load the trained model from the local file system
    lesson2_model = tf.keras.models.load_model(local_model_path, compile=False)
    st.write("Model loaded successfully.")  # Debug statement
except Exception as e:
    # Display error message if model loading fails
    st.error(f"Error loading the model: {e}")
    st.error(f"Exception traceback: {traceback.format_exc()}")
    st.stop()


# Define a custom video processor class inheriting from VideoProcessorBase
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = lesson2_model  # Initialize model attribute with loaded model
        self.actions = np.array(['bad', 'can', 'candy', 'done', 'dont_like', 'dont_understand', 'i_love_you', 'love',
                                 'mom', 'more'])  # Define action labels
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
                    self.sentence.append(self.actions[predicted_action_index])  # Append predicted action to sentence list

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


def lesson_page_2():
    st.title("Lesson 2")
    st.write("Select any of the gestures you'd like to see. Deselect them if you no longer need them. When you are "
             "ready, select 'Start Camera' to begin practicing the gestures. Remember to go slow and try a few times.")
    st.write(" In this lesson, we will practice the following gestures:")

    # Define a dictionary mapping gesture names to GIF paths
    gesture_gifs = {
        "bad": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExdmRkMHVha3owcGh6ZTdjcjFmN2phNGQ3eHBuZzBhdzF3NW16bWcyMCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/5mnugOebs8QfEAumvp/giphy.gif",
        "can": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExZG9rb3R3bGZmNjR0ZDBmNXpjMGV6cjd0eW4wOTh5bjd6ZjgzZ2E2byZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/Zg23eyIgfKqaz01OAf/giphy.gif",
        "candy": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExcGR3cWMybXN0NjlyeDRybjV4N2c1bWJjaGxrdmlvY2hldjR6NWppMiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/MBHaV3hS3Fq70xyWxF/giphy.gif",
        "done": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExa2dlZmJydDk4NnE0MHowNmxmemU4dmZsaGtjeTRncHhoNHMycTJtbyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/D8ALXSR5OFwUymdv4H/giphy.gif",
        "don't like": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExczMxc21nOTljd2lvZHliZ3A3ODlndm1tejFsaGRkdGd0azE4b2JwNSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/vC26YjSBACdIxpSLwl/giphy.gif",
        "don't understand": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExNDZxN3Q4a2h2cnRtaWdnOHM4dnZhd3g2N2swZXo2cXk4a2g3cHdkOSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/6FMnR3KHHC0MjNcvwD/giphy.gif",
        "I love you": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExZnVkNml4cndobDF4aTJscW9vNnZ5cW5vYnlnOWdsYnFwaDdqdXRsaiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/VmdcaKXCILycLK4HKx/giphy.gif",
        "love": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExcmtpbGxoMmt0d2ExdGw1dTFsMzJndXRraDJmemdkN211ZDB4dG0wZCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3O4yBhGy6i6YrPi7Lk/giphy.gif",
        "mom": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExanZ6aDc4M283OXo0dHJteW9qN3NoZ2ZjejdrZHJyZmNtcXJqNHR3dCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/NJYZlVbecz0PoHvNLo/giphy.gif",
        "more": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExOHoyczRseGoxaGplMjNzZGIzYWt2M2xjdG42Mjg3eHVvNzZ6YXUzZiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/rNjL6EDJdultC1B6dN/giphy.gif"
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