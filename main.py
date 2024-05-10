import os
import streamlit as st
from lesson1 import lesson_page_1
from lesson2 import lesson_page_2
from lesson3 import lesson_page_3
from lesson4 import lesson_page_4

def information_page():
    st.title("Hands On Learning")
    st.write("Welcome to Hands on Learning's virtual lesson practice. Please select a lesson from the left "
             "section of the page. You will be able to use your webcam to practice your ASL gestures for each "
             "of the lessons.")

    st.write("The program will start predicting gestures as soon as the video feed begins, analyzing each frame for "
             "possible gestures. Take your time and try the gestures multiple times for accurate feedback.")

def main():
    st.sidebar.title("Hands On Learning")
    lesson = st.sidebar.selectbox("Select Lesson", ["Home", "Lesson 1", "Lesson 2", "Lesson 3", "Lesson 4"])

    if lesson == "Home":
        information_page()
    elif lesson == "Lesson 1":
        lesson_page_1()
    elif lesson == "Lesson 2":
        lesson_page_2()
    elif lesson == "Lesson 3":
        lesson_page_3()
    elif lesson == "Lesson 4":
        lesson_page_4()

    # Check if lesson1.keras file exists
    lesson1_exists = os.path.exists('lesson1.keras')

    # Check if lesson2.keras file exists
    lesson2_exists = os.path.exists('lesson2.keras')

    # Check if lesson3.keras file exists
    lesson3_exists = os.path.exists('lesson3.keras')

    # Check if lesson4.keras file exists
    lesson4_exists = os.path.exists('lesson4.keras')

    # Display messages based on file existence
    st.title("Check for Keras Model Files")

    if lesson1_exists:
        st.write("lesson1.keras file found.")
    else:
        st.error("lesson1.keras file not found.")

    if lesson2_exists:
        st.write("lesson2.keras file found.")
    else:
        st.error("lesson2.keras file not found.")

    if lesson3_exists:
        st.write("lesson3.keras file found.")
    else:
        st.error("lesson3.keras file not found.")

    if lesson4_exists:
        st.write("lesson4.keras file found.")
    else:
        st.error("lesson4.keras file not found.")

if __name__ == "__main__":
    main()
