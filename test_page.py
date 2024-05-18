import streamlit as st
import os
import tensorflow as tf
import traceback

from gcsfs import GCSFileSystem

# Initialize a GCS file system object
fs = GCSFileSystem(project='keras-file-storage')

# Specify the path to the model file in the GCS bucket
model_path = 'gs://keras-files/lesson1.keras'
local_model_path = 'lesson1.keras'

# Download the model file from GCS to local file system
with fs.open(model_path, 'r') as f_in:
    with open(local_model_path, 'w') as f_out:
        f_out.write(f_in.read())


def load_file(local_model_path):
    # models_dir = "models"
    # file_path = os.path.join(os.path.dirname(__file__), models_dir, file_name)
    # try:
    #     with open(file_path, "r") as file:
    #         file_contents = file.read()
    #     st.write("File Contents:")
    #     st.write(file_contents)
    # except FileNotFoundError:
    #     st.error(f"File not found: {file_path}")

    # Load the model outside the function
    try:
        # Load the model directly using tf.keras
        lesson1_model = tf.keras.models.load_model(local_model_path, compile=False)
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        st.error(f"Exception traceback: {traceback.format_exc()}")
        st.stop()


def check_file_exists(file_path):
    if os.path.isfile(file_path):
        st.success(f"File found: {file_path}")
    else:
        st.error(f"File not found: {file_path}")


def test_page():
    st.title("Test Page")
    st.write("Click the button below to load a file from the 'models' directory:")
    if st.button("Load File"):
        load_file(local_model_path)
        st.write("Done")

    st.title("File Existence Checker")

    # Check if the file exists
    check_file_exists(local_model_path)



