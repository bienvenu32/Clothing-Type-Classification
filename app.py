from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown
import h5py

app = Flask(__name__)

# Model details
MODEL_PATH = "clothing_model_converted.h5"
GDRIVE_URL = "https://drive.google.com/file/d/124c2uxs3dzBTjqzSLuPk71eyrqofhoc9/view?usp=sharing"

# Ensure model is downloaded and exists
def download_model():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1024:
        print("Downloading model from Google Drive...")
        gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
        print("Model downloaded successfully.")

download_model()

# Load the trained model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Define class labels
class_names = ['pants', 'shirt', 'shoes', 'shorts', 'sneakers', 't-shirt']

# Image preprocessing function
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize for model input
    image = np.array(image) / 255.0   # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        image = Image.open(file).convert('RGB')
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        confidence = np.max(predictions)

        if confidence < 0.9:  # Confidence threshold
            predicted_class = "unknown type of cloth"
        else:
            predicted_class = class_names[np.argmax(predictions)] if np.argmax(predictions) < len(class_names) else "unknown type of cloth"

        return jsonify({'class': predicted_class, 'confidence': float(confidence)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
