import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, WebRtcMode


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

        # Draw text overlay
        if len(self.sentence) > 0:
            image = self.draw_text_overlay(image, self.sentence[-1])

        return image

    def draw_text_overlay(self, image, text):
        cv2.putText(image, text, (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        return image


# Create an instance of the VideoTransformerWithTextOverlay class
video_transformer = VideoTransformerWithTextOverlay()

# Start WebRTC streaming with the custom video transformer
webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, video_transformer_factory=video_transformer)
