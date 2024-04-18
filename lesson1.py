import streamlit as st
import cv2
import numpy as np
import os
import mediapipe as mp
from tensorflow.keras.models import load_model

from shared_functions import mediapipe_detection, extract_key_points

mp_holistic = mp.solutions.holistic


def lesson_page_1():
    st.title("Lesson 1")
    st.write("In this lesson, we will practice the gestures 'again', 'alive', 'dad', 'family', 'friend', "
             "'hard_of_hearing', 'help_me', 'how', 'hungry', and 'like'")

    try:
        lesson1_model = load_model('lesson1.keras')
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        st.stop()

    # Sets path for exported data (numpy arrays)
    DATA_PATH = os.path.join('lesson1')

    # Actions to detect (13 actions multiplied by 30 frames multiplied by 30 sequences)
    lesson1_actions = np.array(['again', 'alive', 'dad', 'family', 'friend', 'hard_of_hearing', 'help_me', 'how',
                                'hungry', 'like'])

    # Number of videos
    num_sequences = 30

    # Number of frames
    sequence_length = 30

    # Create a dictionary of labels
    lesson1_label_map = {label:num for num, label in enumerate(lesson1_actions)}

    # Array of sequences (features) used to train model to represent relationship between labels
    lesson1_sequences, lesson1_labels = [], []

    # Loop through each action
    for action in lesson1_actions:

        # Loop through each sequence
        for sequence_index in range(num_sequences):

            # Blank array to represent all frames for particular sequence
            window = []

            # Loop through each frame
            for frame_num in range(sequence_length):
                # Load frame
                res = np.load(os.path.join(DATA_PATH, action, str(sequence_index), "{}.npy".format(frame_num)))

                # Add frames to window
                window.append(res)

            # Append video to sequences
            lesson1_sequences.append(window)

            # Append labels
            lesson1_labels.append(lesson1_label_map[action])

    # Function to start the video feed
    def start_video_feed():

        # Button to stop the video feed
        stop_button_pressed = st.button("Stop camera")

        # Appending to list collects 30 frames to generate prediction
        sequence = []

        # Allows concatenation of history
        sentence = []

        predictions = []

        # Only renders results if above a certain threshold
        threshold = 0.4

        # Open the video capture device
        capture = cv2.VideoCapture(0)  # Use 0 for the default camera, or change it to 1 if you have multiple cameras

        # Display a placeholder for the frame
        frame_placeholder = st.empty()

        # Initial detection confidence & tracking confidence set
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

            # Loop to continuously read frames until the stop button is pressed or the video feed ends
            while capture.isOpened() and not stop_button_pressed:
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
                    res = lesson1_model.predict(np.expand_dims(sequence, axis=0))[0]
                    predicted_action_index = np.argmax(res)
                    predictions.append(predicted_action_index)


                    # Visualization logic
                    # If result above threshold
                    if results[predicted_action_index] > threshold:
                        sentence.append(lesson1_actions[predicted_action_index])

                # If the sentence length is greater than 5
                if len(sentence) > 5:
                    # Grab the last five values
                    sentence = sentence[-5:]

                    # Get the latest predicted action
                    latest_predicted_action = sentence[-1]

                    # Render predictions onto the video frame
                    cv2.putText(image, latest_predicted_action, (3, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

                # Convert the OpenCV image to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Display the frame with predictions overlaid using Streamlit
                frame_placeholder.image(image_rgb, channels="RGB")

                if not ret:  # Check if frame was successfully read
                    st.write("The video capture has ended.")
                    break

                # Check if the stop button is pressed
                if stop_button_pressed:
                    break

                # Check for user input to exit the loop
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Release the video capture device and close all windows
            capture.release()
            cv2.destroyAllWindows()

    # Button to start the video feed
    start_button_pressed = st.button("Start camera")

    # Check if the start button is pressed
    if start_button_pressed:
        start_video_feed()
