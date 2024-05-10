import os
from tensorflow.keras.models import load_model


def load_lesson_model(lesson_number):
    # Get the absolute path to the models directory
    models_dir = os.path.join(os.path.dirname(__file__), 'models')

    # Construct the absolute file path to the model file
    model_file_path = os.path.join(models_dir, f'lesson{lesson_number}.keras')

    # Load the model using the absolute file path
    return load_model(model_file_path)
