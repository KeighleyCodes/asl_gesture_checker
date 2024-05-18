import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from shared_functions import mediapipe_detection, extract_key_points

mp_holistic = mp.solutions.holistic

# Load the pre-trained model
lesson1_model = load_model('lesson1.keras', compile=False)

lesson1_actions = np.array(['again', 'alive', 'dad', 'family', 'friend', 'hard_of_hearing', 'help_me', 'how',
                                    'hungry', 'like'])

def start_video_feed():

    sequence = []
    sentence = []
    predictions = []
    threshold = 0.4

    # Accesses mediapipe model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

        # Open the camera feed
        capture = cv2.VideoCapture(0)

        # While the camera is opened
        while capture.isOpened():
            # Read feed
            ret, frame = capture.read()

            # Make detections
            image, results = mediapipe_detection(frame, holistic)

            # Extract keypoints from video
            keypoints = extract_key_points(results)

            # Append keypoints to sequence list
            sequence.append(keypoints)

            # Keep only the last 30 frames to generate a prediction
            sequence = sequence[-30:]

            # Run prediction only if the length of sequence equals 30
            if len(sequence) == 30:
                try:
                    res = lesson1_model.predict(np.expand_dims(sequence, axis=0))[0]
                    predicted_action_index = np.argmax(res)
                    predictions.append(predicted_action_index)

                    # If result is above threshold
                    if res[predicted_action_index] > threshold:
                        sentence.append(lesson1_actions[predicted_action_index])

                except Exception as e:
                    st.error(f"Error during prediction: {e}")

            # If the sentence length is greater than 5
            if len(sentence) > 5:
                # Grab the last five values
                sentence = sentence[-5:]

            # Display the last recognized action on the frame
            if len(sentence) > 0:
                cv2.putText(image, sentence[-1], (3, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

            # Display the frame with predictions overlaid using Streamlit
            st.image(image, channels="BGR", use_column_width=True)

            # Check if 'q' is pressed to exit
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        # Release the camera feed and close all windows
        capture.release()
        cv2.destroyAllWindows()

# Button to start the camera feed
start_button_pressed = st.button("Start Camera")

# Start camera feed when button is pressed
if start_button_pressed:
    start_video_feed()