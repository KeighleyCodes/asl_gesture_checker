import streamlit as st
import os


def load_file(file_name):
    models_dir = "models"
    file_path = os.path.join(os.path.dirname(__file__), models_dir, file_name)
    try:
        with open(file_path, "r") as file:
            file_contents = file.read()
        st.write("File Contents:")
        st.write(file_contents)
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")


def test_page():
    st.title("Test Page")
    st.write("Click the button below to load a file from the 'models' directory:")
    if st.button("Load File"):
        load_file("example.txt")
