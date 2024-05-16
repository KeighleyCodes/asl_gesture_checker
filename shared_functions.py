import cv2
import numpy as np
import requests
import streamlit as st
from tensorflow.python.keras.models import load_model


# Detection function
def mediapipe_detection(image, model):
    # Converts BGR to RGB color
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Sets image to non-writeable status
    image.flags.writeable = False

    # Makes detection
    results = model.process(image)

    # Sets image back to writeable status
    image.flags.writeable = True

    # Convert back to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Return image and results back to loop
    return image, results


# Function to extract key points, if none found creates array of zeros
def extract_key_points(results):
    # Extracts pose key points into one array, if none found creates array of zeros to copy shape for error handling
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)

    # Extracts face key points, if none found creates array of zeros to copy shape for error handling
    face = np.array([[res.x, res.y, res.z] for res in
                     results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)

    # Extracts left hand key points, if none found creates array of zeros to copy shape for error handling
    left_hand = np.array([[res.x, res.y, res.z] for res in
                          results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks \
        else np.zeros(21 * 3)

    # Extracts right hand landmarks, if none found creates array of zeros to copy shape for error handling
    right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(21 * 3)

    return np.concatenate([pose, face, left_hand, right_hand])


def display_gesture_checkboxes(gesture_gifs):
    # Define the number of columns for checkboxes
    num_cols = 3

    # Define a checkbox for each gesture
    selected_gestures = {}
    gesture_columns = [st.columns(num_cols) for _ in range((len(gesture_gifs) + num_cols - 1) // num_cols)]
    for i, gesture_name in enumerate(gesture_gifs.keys()):
        col_index = i // num_cols
        selected_gestures[gesture_name] = gesture_columns[col_index][i % num_cols].checkbox(gesture_name)

    # Return the dictionary of selected gestures
    return selected_gestures


# Function to display GIF gestures
def display_gif(gif_path, gesture_name):
    # Display the GIF
    st.markdown(f"![Gesture GIF]({gif_path})")

    # Store the state of the displayed GIF
    st.session_state[f"{gesture_name}_gif_displayed"] = True


# Function to download keras files
# def download_file(url, local_filename):
#     with requests.get(url, stream=True) as r:
#         r.raise_for_status()
#         with open(local_filename, 'wb') as f:
#             for chunk in r.iter_content(chunk_size=8192):
#                 f.write(chunk)

def load_model_from_google_drive(google_drive_link):
    file_id = google_drive_link.split('/')[-2]
    download_link = f'https://drive.google.com/uc?export=download&id={file_id}'
    response = requests.get(download_link)
    with open('model.h5', 'wb') as f:
        f.write(response.content)
    return tf.keras.models.load_model('model.h5')