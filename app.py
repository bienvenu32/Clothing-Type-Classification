from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("clothing_model.h5")
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
    app.run(debug=True)
