import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import traceback
from shared_functions import mediapipe_detection, extract_key_points, display_gif, display_gesture_checkboxes

mp_holistic = mp.solutions.holistic

# Load model
try:
    lesson3_model = load_model('lesson3.keras')
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.error(f"Exception traceback: {traceback.format_exc()}")
    st.stop()

# Define actions
lesson3_actions = np.array(['name', 'need', 'now', 'please', 'sad', 'school', 'slow', 'sorry', 'wait', 'want'])


# Function to start the video feed
def start_video_feed3(model):
    # Button to stop the video feed
    stop_button_pressed = st.button("Stop camera")

    # Appending to list collects 30 frames to generate prediction
    sequence = []

    # Allows concatenation of history
    sentence = []

    predictions = []

    # Only renders results if above a certain threshold
    threshold = 0.4

    # Function for opening the video feed
    capture = cv2.VideoCapture(0)

    # Display a placeholder for the frame
    frame_placeholder = st.empty()

    # Initial detection confidence & tracking confidence set
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        # While the camera is opened
        while capture.isOpened():
            # Reads feed
            ret, frame = capture.read()

            if not ret:
                st.write("The video capture has ended or failed.")
                break

            # Make detections
            image, results = mediapipe_detection(frame, holistic)

            # Extract key points from video
            key_points = extract_key_points(results)

            # Appending key points to sequence list
            sequence.append(key_points)

            # Grabs the last 30 frames to generate a prediction
            sequence = sequence[-30:]

            # Run prediction only if the length of sequence equals 30
            if len(sequence) == 30:
                try:
                    results = model.predict(np.expand_dims(sequence, axis=0))[0]
                    predicted_action_index = np.argmax(results)
                    predictions.append(predicted_action_index)

                    if results[predicted_action_index] > threshold:
                        sentence.append(lesson3_actions[predicted_action_index])
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
                    st.error(f"Exception traceback: {traceback.format_exc()}")

            if len(sentence) > 5:
                sentence = sentence[-5:]

            if len(sentence) > 0:
                cv2.putText(image, sentence[-1], (3, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

            # Convert the OpenCV image to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Display the frame with predictions overlaid using Streamlit
            frame_placeholder.image(image_rgb, channels="RGB")

            # Check if the stop button is pressed
            if stop_button_pressed:
                break

    # Releases the camera feed, closes all windows
    capture.release()
    cv2.destroyAllWindows()

def lesson_page_3():
    st.title("Lesson 3")
    st.write("Select any of the gestures you'd like to see. Deselect them if you no longer need them. When you are "
             "ready, select 'Start Camera' to begin practicing the gestures. Remember to go slow and try a few times.")
    st.write("In this lesson, we will practice the following gestures:")

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

    # Checkboxes for selecting GIFs
    selected_gestures = display_gesture_checkboxes(gesture_gifs)
    for gesture_name, selected in selected_gestures.items():
        if selected:
            display_gif(gif_path=gesture_gifs[gesture_name], gesture_name=gesture_name)

    # Button to start the video feed
    start_button_pressed = st.button("Start camera")

    if start_button_pressed:
        # Start the video feed
        start_video_feed3(lesson3_model)
