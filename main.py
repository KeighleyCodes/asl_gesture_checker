import streamlit as st
from lesson1 import lesson_page_1
from lesson2 import lesson_page_2
from lesson3 import lesson_page_3
from lesson4 import lesson_page_4


def information_page():
    st.title("Hands On Learning")
    st.write("Welcome to Hands on Learning's virtual lesson practice. Please select a lesson from the left section of "
             "the page. You will be able to use your webcam to practice your ASL gestures for each of the lessons.")
    st.write("The program will begin analyzing the video feed once it detects both hands in the frame. There may be a "
             "slight delay before it starts displaying the predicted gestures. Once initiated, the program will swiftly "
             "predict gestures frame by frame as you move, providing immediate feedback. Take your time and repeat the "
             "gestures as needed for accurate analysis")


def main():
    st.empty()
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


if __name__ == "__main__":
    main()
