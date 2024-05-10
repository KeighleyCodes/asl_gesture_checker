import streamlit as st
import cv2
import numpy as np
import os
import mediapipe as mp
from tensorflow.keras.models import load_model
import traceback

from shared_functions import mediapipe_detection, extract_key_points, display_gif, display_gesture_checkboxes

mp_holistic = mp.solutions.holistic


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

    selected_gestures = display_gesture_checkboxes(gesture_gifs)
    for gesture_name, selected in selected_gestures.items():
        if selected:
            display_gif(gif_path=gesture_gifs[gesture_name], gesture_name=gesture_name)

    # Button to start the video feed
    start_button_pressed = st.button("Start camera")

    if start_button_pressed:
        try:
            lesson2_model = load_model('lesson2.keras')
        except Exception as e:
            st.error(f"Error loading the model: {e}")
            st.error(f"Exception traceback: {traceback.format_exc()}")
            st.stop()

        # Sets path for exported data (numpy arrays)
        DATA_PATH = os.path.join('lesson2')

        # Actions to detect (x actions multiplied by 30 frames multiplied by 30 sequences)
        lesson2_actions = np.array(
            ['bad', 'can', 'candy', 'done', 'dont_like', 'dont_understand', 'i_love_you', 'love', 'mom', 'more'])

        # Number of videos
        num_sequences = 30

        # Number of frames
        sequence_length = 30

        # Creates a dictionary of labels
        lesson2_label_map = {label: num for num, label in enumerate(lesson2_actions)}

        # Array of sequences (features) used to train model to represent relationship between labels
        lesson2_sequences, lesson2_labels = [], []

        # loops through each action
        for action in lesson2_actions:

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
                lesson2_sequences.append(window)

                # Append labels
                lesson2_labels.append(lesson2_label_map[action])

        # Function to start the video feed
        def start_video_feed2():

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
                        results = lesson2_model.predict(np.expand_dims(sequence, axis=0))[0]
                        predicted_action_index = np.argmax(results)
                        predictions.append(predicted_action_index)

                        # Visualization logic
                        # If result above threshold
                        if results[predicted_action_index] > threshold:
                            sentence.append(lesson2_actions[predicted_action_index])

                    # If the sentence length is greater than 5
                    if len(sentence) > 5:
                        # Grab the last five values
                        sentence = sentence[-5:]

                    if len(sentence) > 0:
                        cv2.putText(image, sentence[-1], (3, 30),
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

                # Releases the camera feed, closes all windows
                capture.release()
                cv2.destroyAllWindows()

        # Start the video feed
        start_video_feed2()
