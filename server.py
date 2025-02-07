from flask import Flask, request, jsonify, send_file
import base64
import numpy as np
import cv2
import io
from ai import recognize_and_translate

app = Flask(__name__)


@app.route("/")
def home():
    return send_file("index.html")


@app.route("/recognize", methods=["POST"])
def recognize():
    # Get the image data from the request
    data = request.json
    image_data = data["image"].split(",")[1]

    # Convert base64 to image
    image_bytes = base64.b64decode(image_data)
    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)

    # Enhanced preprocessing
    image = cv2.GaussianBlur(image, (5, 5), 0)
    image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)[1]

    # Save temporary image
    temp_path = "temp_digit.png"
    cv2.imwrite(temp_path, image)

    # Get recognition result
    result = recognize_and_translate(temp_path)

    return jsonify({"result": result})


if __name__ == "__main__":
    app.run(debug=True)
