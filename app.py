from flask import Flask, request, render_template, jsonify
import numpy as np
import cv2
import os
import joblib

app = Flask(__name__)

# Ensure the model directory exists
MODEL_PATH = os.path.join("model", "model.pkl")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

# Load the trained model
model = joblib.load(MODEL_PATH)

# Define class labels
CLASS_NAMES = [ "Early Blight", "Healthy", "Late Blight"]

# Image size used during training
IMAGE_SIZE = (64, 64)

def preprocess_image(image_file):
    # Read and preprocess the uploaded image
    img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, IMAGE_SIZE)
    img = img.flatten() / 255.0  # Flatten and normalize
    img = np.expand_dims(img, axis=0)  # Make it batch-like
    return img

@app.route('/')
def home():
    return render_template('index.html')  # Ensure index.html exists in templates/

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    try:
        img = preprocess_image(file)
        prediction = model.predict_proba(img)

        class_index = int(np.argmax(prediction))
        class_name = CLASS_NAMES[class_index]
        confidence = float(np.max(prediction))

        return jsonify({'class': class_name, 'confidence': round(confidence, 4)})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)