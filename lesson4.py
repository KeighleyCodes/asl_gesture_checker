import requests
import streamlit as st
import cv2
import numpy as np
import os
import mediapipe as mp
from tensorflow.keras.models import load_model
import traceback
from shared_functions import mediapipe_detection, extract_key_points, display_gif, display_gesture_checkboxes

mp_holistic = mp.solutions.holistic

# GCS URL where models are stored
gcs_base_url = "https://storage.googleapis.com/my-keras-files-bucket/"


def download_file(url, local_filename):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


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

    # Checkboxes for selecting GIFs
    selected_gestures = display_gesture_checkboxes(gesture_gifs)
    for gesture_name, selected in selected_gestures.items():
        if selected:
            display_gif(gif_path=gesture_gifs[gesture_name], gesture_name=gesture_name)

    # Button to start the video feed
    start_button_pressed = st.button("Start camera")

    if start_button_pressed:

        # Load model
        try:
            # Download the model file
            lesson4_model_url = os.path.join(gcs_base_url, 'lesson4.h5')
            local_model_path = 'lesson4.h5'
            download_file(lesson4_model_url, local_model_path)

            # Load model from local file
            lesson4_model = load_model(local_model_path)
        except Exception as e:
            st.error(f"Error loading the model: {e}")
            st.error(f"Exception traceback: {traceback.format_exc()}")
            st.stop()

        # Sets path for exported data (numpy arrays)
        DATA_PATH = os.path.join('lesson4')

        # Actions to detect (11 actions multiplied by 30 frames multiplied by 30 sequences)
        lesson4_actions = np.array(['good', 'happy', 'hearing', 'mine', 'no', 'yes', 'what', 'where', 'who', 'you',
                                    'yours'])

        # Number of videos
        num_sequences = 30

        # Number of frames
        sequence_length = 30

        # Creates a dictionary of labels
        lesson4_label_map = {label: num for num, label in enumerate(lesson4_actions)}

        # Array of sequences (features) used to train model to represent relationship between labels
        lesson4_sequences, lesson4_labels = [], []

        # loops through each action
        for action in lesson4_actions:

            # Loops through each sequence
            for sequence_index in range(num_sequences):

                # Blank array to represent all frames for particular sequence
                window = []

                # Loops through each frame
                for frame_num in range(sequence_length):
                    # Loads frame
                    res = np.load(os.path.join(DATA_PATH, action, str(sequence_index), "{}.npy".format(frame_num)))

                    # Add frames to window
                    window.append(res)

                # Append video to sequences
                lesson4_sequences.append(window)

                # Append labels
                lesson4_labels.append(lesson4_label_map[action])

        # Function to start the video feed
        def start_video_feed4():

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
                            results = lesson4_model.predict(np.expand_dims(sequence, axis=0))[0]
                            predicted_action_index = np.argmax(results)
                            predictions.append(predicted_action_index)

                            if results[predicted_action_index] > threshold:
                                sentence.append(lesson4_actions[predicted_action_index])
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

                    # Check if frame was successfully read
                    if not ret:
                        st.write("The video capture has ended.")
                        break

                    # Check if the stop button is pressed
                    if stop_button_pressed:
                        break

            # Releases the camera feed, closes all windows
            capture.release()
            cv2.destroyAllWindows()

        # Start the video feed
        start_video_feed4()
