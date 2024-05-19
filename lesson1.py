import gcsfs
import streamlit as st
import cv2
import numpy as np
import os
import mediapipe as mp
import tensorflow as tf
from gcsfs import GCSFileSystem
from keras.src.export.export_lib import TFSMLayer
from tensorflow.keras.models import load_model
import traceback
from shared_functions import mediapipe_detection, extract_key_points, display_gif, display_gesture_checkboxes

mp_holistic = mp.solutions.holistic

# Initialize a GCS file system object
fs = GCSFileSystem(project='keras-file-storage')

# Specify the path to the model file in the GCS bucket
model_path = 'gs://keras-files/lesson1.keras'
local_model_path = 'lesson1.keras'

# Download the model file from GCS to local file system
with fs.open(model_path, 'rb') as f_in:
    with open(local_model_path, 'wb') as f_out:
        f_out.write(f_in.read())
        f_out.flush()
        f_out.close()


# Load the model outside the function
try:
    # Load the model directly using tf.keras
    lesson1_model = tf.keras.models.load_model(local_model_path, compile=False)
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.error(f"Exception traceback: {traceback.format_exc()}")
    st.stop()


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
        # Sets path for exported data (numpy arrays)
        DATA_PATH = os.path.join('lesson1')
        local_data_path = 'lesson1'

        lesson1_actions = np.array(['again', 'alive', 'dad', 'family', 'friend', 'hard_of_hearing', 'help_me', 'how',
                                    'hungry', 'like'])

        num_sequences = 30

        sequence_length = 30

        lesson1_label_map = {label: num for num, label in enumerate(lesson1_actions)}

        lesson1_sequences, lesson1_labels = [], []

        # Ensure local directories exist
        os.makedirs(local_data_path, exist_ok=True)

        for action in lesson1_actions:
            os.makedirs(os.path.join(local_data_path, action), exist_ok=True)

            for sequence_index in range(num_sequences):
                os.makedirs(os.path.join(local_data_path, action, str(sequence_index)), exist_ok=True)

                window = []

                for frame_num in range(sequence_length):
                    os.makedirs(os.path.join(local_data_path, action, str(sequence_index)), exist_ok=True)

                    path_to_load = os.path.join(DATA_PATH, action, str(sequence_index), "{}.npy".format(frame_num))
                    local_path_to_load = os.path.join(local_data_path, action, str(sequence_index), f"{frame_num}.npy")

                    with fs.open(path_to_load, 'rb') as f_in:
                        with open(local_path_to_load, 'wb') as f_out:
                            f_out.write(f_in.read())

                    res = np.load(path_to_load)
                    st.write(path_to_load)

                    window.append(res)

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
                        try:
                            results = lesson1_model.predict(np.expand_dims(sequence, axis=0))[0]
                            predicted_action_index = np.argmax(results)
                            predictions.append(predicted_action_index)

                            if results[predicted_action_index] > threshold:
                                sentence.append(lesson1_actions[predicted_action_index])
                        except Exception as e:
                            st.error(f"Error during prediction: {e}")
                            st.error(f"Exception traceback: {traceback.format_exc()}")

                    if len(sentence) > 5:
                        sentence = sentence[-5:]

                    if len(sentence) > 0:
                        cv2.putText(image, sentence[-1], (3, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    frame_placeholder.image(image_rgb, channels="RGB")

                    if not ret:
                        st.write("The video capture has ended.")
                        break

                    if stop_button_pressed:
                        break

            capture.release()
            cv2.destroyAllWindows()

        start_video_feed1()