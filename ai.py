import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import os


def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)[1]
    img = cv2.resize(img, (28, 28))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def number_to_text(number):
    numbers_text = {
        0: "zero",
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
    }
    return numbers_text.get(number, "unknown")


# Check if model exists and load it
MODEL_PATH = "handwriting_model.h5"  # Changed from handwriting_model_final.h5

if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file '{MODEL_PATH}' not found!")
    print("Please run the following commands:")
    print("1. python train_model.py")
    print("2. Wait for training to complete")
    print("3. Then run server.py again")
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found")

try:
    model = keras.models.load_model(MODEL_PATH)
    print(f"Model loaded successfully from {os.path.abspath(MODEL_PATH)}")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise


def recognize_and_translate(image_path):
    processed_image = preprocess_image(image_path)
    prediction = model.predict(
        processed_image, verbose=0
    )  # Added verbose=0 to reduce output
    predicted_digit = np.argmax(prediction[0])
    confidence = float(prediction[0][predicted_digit])
    text_result = number_to_text(predicted_digit)
    return f"Recognized digit: {predicted_digit}, Text: {text_result} (Confidence: {confidence:.2%})"
