import streamlit as st
import os


def check_file_exists(file_path):
    if os.path.isfile(file_path):
        st.success(f"File found: {file_path}")
    else:
        st.error(f"File not found: {file_path}")


def test_page_2():
    st.title("File Existence Checker")

    # Define the file path to check
    file_path = "lesson1.keras"

    # Check if the file exists
    check_file_exists(file_path)


