from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import requests

app = Flask(__name__)

# Load the trained model
model_path = "clothing_model.h5"
if not os.path.exists(model_path):
    print("Model file not found locally. Downloading from Google Drive...")
    url = "https://drive.google.com/uc?id=1GV22U9uaXsKN8WGvzTosuqEBr02XMtzV"
    response = requests.get(url)
    with open(model_path, "wb") as f:
        f.write(response.content)
    print("Model downloaded successfully.")

model = tf.keras.models.load_model(model_path)
class_names = ['pants', 'shirt', 'shoes', 'shorts', 'sneakers', 't-shirt']  # Same as in training

def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to match model input
    image = np.array(image) / 255.0  # Normalize pixel values
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
