import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


# Detection function
def mediapipe_detection(image, model):
    # Converts BGR to RGB color
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Sets image to unwriteable status
    image.flags.writeable = False

    # Makes detection
    results = model.process(image)

    # Sets image back to writeable status
    image.flags.writeable = True

    # Convert back to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Return image and results back to loop
    return image, results


def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                              mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))

    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))

    # Draw left-hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))

    # Draw right-hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))


# Function for opening the video feed
capture = cv2.VideoCapture(0)

# Accesses mediapipe model
# Initial detection confidence & tracking confidence set
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    # While camera is opened
    while capture.isOpened():
        # Reads feed
        ret, frame = capture.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)

        # Draw landmarks on live video
        draw_styled_landmarks(image, results)

        # Shows to screen
        cv2.imshow('Open Camera Feed', image)

        # Break when 'q' pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Releases camera feed, closes all windows
    capture.release()
    cv2.destroyAllWindows()


# Function to extract keypoints, if none found creates array of zeros
def extract_keypoints(results):
    # Extracts pose keypoints into one array, if none found creates array of zeros to copy shape for error handling
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)

    # Extracts face keypoints, if none found creates array of zeros to copy shape for error handling
    face = np.array([[res.x, res.y, res.z] for res in
                     results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)

    # Extracts left hand keypoints, if none found creates array of zeros to copy shape for error handling
    left_hand = np.array([[res.x, res.y, res.z] for res in
                          results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(
        21 * 3)

    # Extracts right hand landmarks, if none found creates array of zeros to copy shape for error handling
    right_hand = np.array([[res.x, res.y, res.z] for res in
                           results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
        21 * 3)

    return np.concatenate([pose, face, left_hand, right_hand])


draw_styled_landmarks(frame, results)
result_test = extract_keypoints(results)
result_test
np.save('0', result_test)
np.load('0.npy')
plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
# Sets path for exported data (numpy arrays)
DATA_PATH = os.path.join('MP-ASL-gestures')

# Actions to detect (x actions multiplied by 30 frames multiplied by 30 sequences)
actions = np.array(
    ['again', 'deaf', 'family', 'good', 'hard_of_hearing', 'hearing', 'inside', 'like', 'name', 'no', 'sorry',
     'thank_you', 'what'])

# Number of videos
num_sequences = 30

# Number of frames
sequence_length = 30
# Loops through all actions and makes directories for them
for action in actions:
    # loop through videos
    for sequence in range(num_sequences):
        # Creates new folders and makes directories with subfolders, skips if folder already created
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass
# Function for opening the video feed
capture = cv2.VideoCapture(0)

# Accesses mediapipe model
# Initial detection confidence & tracking confidence set
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    # Loop through actions
    for action in actions:

        # Loop though sequences
        for sequence in range(num_sequences):

            # Loop through video length
            for frame_num in range(sequence_length):

                # Reads feed
                ret, frame = capture.read()

                # Make detections
                image, results = mediapipe_detection(frame, holistic)

                # Draw landmarks on live video
                draw_styled_landmarks(image, results)

                # Provides break for repositioning on first frame

                # If frame is first frame
                if frame_num == 0:
                    # Outprints to screen to show frames being collected
                    cv2.putText(image, 'COLLECTION STARTING', (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4,
                                cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video number {}'.format(action, sequence), (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                    # Take 2 second break between each video
                    cv2.waitKey(2000)

                else:
                    # Prints out image currently collecting
                    cv2.putText(image, 'Collecting frames for {} Video number {}'.format(action, sequence), (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                # Exports keypoints
                keypoints = extract_keypoints(results)

                # Creates path
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))

                # Saves keypoints
                np.save(npy_path, keypoints)

                # Shows to screen
                cv2.imshow('Open Camera Feed', image)

                # Break when 'q' pressed
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

    # Releases camera feed, closes all windows
    capture.release()
    cv2.destroyAllWindows()
# Allows training into partitions for training and testing
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Creates a dictionary of labels
label_map = {label: num for num, label in enumerate(actions)}
label_map
# Array of sequences (features) used to train model to represent relationship between labels
sequences, labels = [], []

# loops through each action
for action in actions:

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
        sequences.append(window)

        # Append lables
        labels.append(label_map[action])
np.array(sequences).shape
np.array(labels).shape
# Stores sequences in array to make it easier to work with, uses x for x / y values
x = np.array(sequences)
x.shape
# Converts labels to one encoded for y values
y = to_categorical(labels).astype(int)
# Unpack results of train_test_split function with 5% of data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)
# Imports dependancies for training model

# Allows building of sequential neural network
from tensorflow.keras.models import Sequential

# Imports LSTM layer, provides temporal component, allows action detection
from tensorflow.keras.layers import LSTM, Dense

# Imports TensorBoard for logging and tracing during training
from tensorflow.keras.callbacks import TensorBoard

# Imports optimizer
from tensorflow.keras.optimizers import Adam

# Create log directory
log_dir = os.path.join('Log-dir')

# Set up TensorBoard callback
tb_callback = TensorBoard(log_dir=log_dir)
# Instantiates model
model = Sequential()

# Adds three sets of LSTM layers
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))

# Adds dense fully connected layers for fully connected neural network
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))
# Set up model for training and metrics for tracking
optimizer = Adam(learning_rate=0.0001)  # Set a lower learning rate
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
# Train model
model.fit(x_train, y_train, epochs=500, callbacks=[tb_callback])
model.summary()
results = model.predict(x_test)
actions[np.argmax(results[3])]
actions[np.argmax(y_test[3])]
# Save model
model.save('action6.keras')
# Import confusion matrix to evaluate true and false positives and negatives
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

# Make more predictions for testing accuracy
yhat = model.predict(x_train)

# Extract predicted classes converted to categorical label
ytrue = np.argmax(y_train, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()
# Returns confusion matrix of shape (2,2)
multilabel_confusion_matrix(ytrue, yhat)
accuracy_score(ytrue, yhat)
# Video feed for testing

# Appending to list collects 30 frames to generate prediction
sequence = []

# Allows concatenation of history
sentence = []

predictions = []

# Only renders results if above a certain threshold
threshold = 0.4

# Function for opening the video feed
capture = cv2.VideoCapture(0)

# Accesses mediapipe model
# Initial detection confidence & tracking confidence set
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    # While the camera is opened
    while capture.isOpened():
        # Reads feed
        ret, frame = capture.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)

        # Draw landmarks on live video
        draw_styled_landmarks(image, results)

        # Extract keypoints from video
        keypoints = extract_keypoints(results)

        # Appending keypoints to sequence list
        sequence.append(keypoints)

        # Grabs the last 30 frames to generate a prediction
        sequence = sequence[-30:]

        # Run prediction only if the length of sequence equals 30
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            predicted_action_index = np.argmax(res)
            predicted_action = actions[predicted_action_index]
            print(predicted_action)
            predictions.append(predicted_action)

            # Visualization logic
            # If result above threshold
            if res[predicted_action_index] > threshold:
                sentence.append(predicted_action)

        # If the sentence length is greater than 5
        if len(sentence) > 5:
            # Grab the last five values
            sentence = sentence[-5:]

            # Get the latest predicted action
            latest_predicted_action = sentence[-1]

            # Display the latest predicted action
            cv2.putText(image, latest_predicted_action, (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # Shows to screen
        cv2.imshow('Open Camera Feed', image)

        # Break when 'q' pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Releases the camera feed, closes all windows
capture.release()
cv2.destroyAllWindows()

import tensorflow as tf

# Clear the TensorFlow session and reset the global TensorFlow graph state
tf.compat.v1.reset_default_graph()
tf.keras.backend.clear_session()
capture.release()
cv2.destroyAllWindows()
