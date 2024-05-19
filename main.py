import streamlit as st
from lesson1 import lesson_page_1
from lesson2 import lesson_page_2
from lesson3 import lesson_page_3
from lesson4 import lesson_page_4
from test_page import test_page


def information_page():

    st.title("Hands On Learning")
    st.write("Welcome to Hands on Learning's virtual lesson practice. Please select a lesson from the left "
             "section of the page. You will be able to use your webcam to practice your ASL gestures for each "
             "of the lessons.")

    st.write("The program will start predicting gestures as soon as the video feed begins, analyzing each frame for "
             "possible gestures. Take your time and try the gestures multiple times for accurate feedback.")


def main():
    # Clear the page
    st.empty()

    # Sidebar title
    st.sidebar.title("Hands On Learning")

    # Sidebar lesson selector box

    lesson = st.sidebar.selectbox("Select Lesson", ["Home", "Lesson 1", "Lesson 2", "Lesson 3", "Lesson 4",
                                                    "FOR TESTING"])

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
    elif lesson == "FOR TESTING":
        test_page()


if __name__ == "__main__":
    main()