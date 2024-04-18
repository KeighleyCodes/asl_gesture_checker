import cv2
import numpy as np


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


