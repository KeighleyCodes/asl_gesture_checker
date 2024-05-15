import os
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import traceback
from st_files_connection import FilesConnection  # Import the connection object
import pandas as pd  # Import pandas for file reading
from shared_functions import display_gif, display_gesture_checkboxes

mp_holistic = mp.solutions.holistic

# Create a connection object for GCS
conn = st.connection('gcs', type=FilesConnection)


# Function to download Keras model file if not already downloaded
def download_keras_model():
    # This function is not needed anymore as we'll load the model from GCS
    pass


# Load Keras model from GCS
def load_keras_model():
    try:
        # Read the Keras model file directly from GCS
        model = load_model(conn.read("lesson1.keras", input_format="keras"))
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        st.error(f"Exception traceback: {traceback.format_exc()}")
        return None


# Main function for lesson page 1
def lesson_page_1():
    st.title("Lesson 1")
    st.write("Select any of the gestures you'd like to see. Deselect them if you no longer need them. When you are "
             "ready, select 'Start Camera' to begin practicing the gestures. Remember to go slow and try a few times.")
    st.write(" In this lesson, we will practice the following gestures:")

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

    start_button_pressed = st.button("Start camera")

    if start_button_pressed:
        # Load Keras model
        model = load_keras_model()
        if model is None:
            return

        # Sets path for exported data (numpy arrays)
        DATA_PATH = os.path.join('lesson1')

        # Check if DATA_PATH exists
        if not os.path.exists(DATA_PATH):
            st.write(f"Directory {DATA_PATH} does not exist.")
        else:
            st.write(f"Directory {DATA_PATH} exists.")

        lesson1_actions = np.array(['again', 'alive', 'dad', 'family', 'friend', 'hard_of_hearing', 'help_me', 'how',
                                    'hungry', 'like'])

        num_sequences = 30
        sequence_length = 30
        lesson1_label_map = {label: num for num, label in enumerate(lesson1_actions)}
        lesson1_sequences, lesson1_labels = [], []

        for action in lesson1_actions:
            for sequence_index in range(num_sequences):
                subdir_path = os.path.join(DATA_PATH, action, str(sequence_index))
                if not os.path.exists(subdir_path):
                    st.write(f"Subdirectory {subdir_path} does not exist.")
                else:
                    st.write(f"Subdirectory {subdir_path} exists.")
                    window = []
                    for frame_num in range(sequence_length):
                        # Replace local file loading with GCS file loading
                        npy_data = conn.read(f"{subdir_path}/{frame_num}.npy", input_format="npy")
                        window.append(npy_data)
                    lesson1_sequences.append(window)
                    lesson1_labels.append(lesson1_label_map[action])

        def start_video_feed1():
            stop_button_pressed = st.button("Stop camera")
            sequence = []
            sentence = []
            predictions = []
            threshold = 0.4
            capture = cv2.VideoCapture(0)
            frame_placeholder = st.empty()
            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                while capture.isOpened():
                    ret, frame = capture.read()
                    image, results = mediapipe_detection(frame, holistic)
                    key_points = extract_key_points(results)
                    sequence.append(key_points)
                    sequence = sequence[-30:]
                    if len(sequence) == 30:
                        results = model.predict(np.expand_dims(sequence, axis=0))[0]
                        predicted_action_index = np.argmax(results)
                        predictions.append(predicted_action_index)
                        if results[predicted_action_index] > threshold:
                            sentence.append(lesson1_actions[predicted_action_index])
                    if len(sentence) > 5:
                        sentence = sentence[-5:]
                    if len(sentence) > 0:
                        cv2.putText(image, sentence[-1], (3, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(image_rgb, channels="RGB")
                    if not ret:  # Check if frame was successfully read
                        st.write("The video capture has ended.")
                        break
                    if stop_button_pressed:
                        break

            capture.release()
            cv2.destroyAllWindows()

        start_video_feed1()


# Run the lesson page function
lesson_page_1()
