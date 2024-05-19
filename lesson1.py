import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import traceback
from gcsfs import GCSFileSystem
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration, VideoTransformerBase
from shared_functions import mediapipe_detection, extract_key_points, display_gif, display_gesture_checkboxes

mp_holistic = mp.solutions.holistic

# Initialize a GCS file system object
fs = GCSFileSystem(project='keras-file-storage')

# Specify the path to the model file in the GCS bucket
model_path = 'gs://keras-files/lesson1.keras'
local_model_path = 'lesson1.keras'

# Download the model file from GCS to local file system
try:
    with fs.open(model_path, 'rb') as f_in:
        with open(local_model_path, 'wb') as f_out:
            f_out.write(f_in.read())
    st.write("Model downloaded successfully.")  # Debug statement
except Exception as e:
    st.error(f"Error downloading the model: {e}")
    st.error(f"Exception traceback: {traceback.format_exc()}")
    st.stop()

# Load the model outside the function
try:
    lesson1_model = tf.keras.models.load_model(local_model_path, compile=False)
    st.write("Model loaded successfully.")  # Debug statement
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.error(f"Exception traceback: {traceback.format_exc()}")
    st.stop()

RTC_CONFIGURATION = {
    "iceServers": [{"urls": "stun:stun.l.google.com:19302"}]
}


class VideoTransformerWithTextOverlay(VideoTransformerBase):
    def __init__(self):
        self.sequence = []
        self.sentence = []
        self.predictions = []
        self.threshold = 0.4
        self.actions = np.array(['again', 'alive', 'dad', 'family', 'friend', 'hard_of_hearing', 'help_me', 'how',
                                 'hungry', 'like'])
        self.model = lesson1_model
        self.mp_holistic = mp.solutions.holistic

    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")

        # Make detections
        image, results = mediapipe_detection(image, self.mp_holistic)

        # Extract key_points from video
        key_points = extract_key_points(results)

        # Appending key_points to sequence list
        self.sequence.append(key_points)

        # Grabs the last 30 frames to generate a prediction
        self.sequence = self.sequence[-30:]

        # Run prediction only if the length of sequence equals 30
        if len(self.sequence) == 30:
            res = self.model.predict(np.expand_dims(self.sequence, axis=0))[0]
            predicted_action_index = np.argmax(res)
            self.predictions.append(predicted_action_index)

            # Visualization logic
            # If result above threshold
            if res[predicted_action_index] > self.threshold:
                self.sentence.append(self.actions[predicted_action_index])

        # If the sentence length is greater than 5
        if len(self.sentence) > 5:
            # Grab the last five values
            self.sentence = self.sentence[-5:]

        # Combine the last 5 predictions into a single string
        prediction_text = ' '.join(self.sentence[-5:])

        # Return the image and prediction text
        return image, prediction_text


# Create an instance of the VideoTransformerWithTextOverlay class
video_transformer = VideoTransformerWithTextOverlay()

# Start WebRTC streaming with the custom video transformer
webrtc_ctx = webrtc_streamer(
    key="example",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    video_transformer_factory=video_transformer,
    async_processing=True,
)

# Layout for displaying the predictions underneath the video stream
st.write("Predictions:")
predictions_placeholder = st.empty()


def lesson_page_1():
    st.title("Lesson 1")
    st.write("Select any of the gestures you'd like to see. Deselect them if you no longer need them. When you are "
             "ready, select 'Start Camera' to begin practicing the gestures. Remember to go slow and try a few times.")
    st.write("In this lesson, we will practice the following gestures:")

    gesture_gifs = {
        "again": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExdWk0ZW1scnI5eHFoMWQ5ZWJiazJuNWQ5OWFtOTRndWo1bHUxdHpyeSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/XSvXpnvUizxUP09NaQ/giphy.gif",
        "alive": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExcXZoenl0MzY2ZGFuOWw5cjVoam93d3FoYzZ1OHd4dGNjbzlrbHNxayZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/FwJwOTJGk1WudLKCSH/giphy.gif",
        "dad": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExY2pxOXFxbDMxbDk5N29sbWx6cnN1NHNpb3ZxejF6ZHJ1dnFmazJsbiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/chRxaZphkmBIsD6lpd/giphy.gif",
        "family": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExZTVheHRzdGVtM3J3YTB2aWl1YmwwcjB1c2tyaTJjcm96bXNpYjI5byZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/46WHQp522QSBgTLRRF/giphy.gif",
        "friend": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExdGU1ZGRhZjdpZWg3c2N1cmJndW1qaHp0ZzQ1dmQ1d2luaWJraXlseSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/0QIxZYF0MBFe0RTrta/giphy.gif",
        "hard of hearing": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExeWlmcWZpZ3FhZ3V4cGt2dXp0ZHpmb3Q4Z3g0amZlNmVyeTk3M3prdiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/k9XFkywBrEiUYReXmp/giphy.gif",
        "help me": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExNHNtaGlwNmVsazUxdnAyZXgzYnExNnFiZmNyNWZ2ZHdxbGg5dXp5NSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/b0Uru1RCENuHq8DhUt/giphy.gif",
        "how": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExM3V3ZGw4cG1iZXAxMzJxNzI3ZzNlbmwwMmIzZWg5ZDJnc2d5dDBrcyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/O6XlVSxTJsvp8LSIg4/giphy.gif",
        "hungry": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExemVzNTlpYnZ3Y3l6ZW53eXcwdnM4MWc2aHFyOGtoanlnbGZnYmgwOSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/VOyExo7liUxLFAeitB/giphy.gif",
        "like": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExbmo3dm41ZDcwYTIzNG01ejkyNjcyYnpzMjRiZnFobTZnYmRkOWczZCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/Laj5J3k8ysyXdLo8m2/giphy.gif"
    }

    selected_gestures = display_gesture_checkboxes(gesture_gifs)
    for gesture_name, selected in selected_gestures.items():
        if selected:
            display_gif(gif_path=gesture_gifs[gesture_name], gesture_name=gesture_name)
