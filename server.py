from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import cv2

app = Flask(__name__)
CORS(app)  # Allow Cross-Origin requests

# Load model
model = load_model('fruit_detection_model.h5')

# Compile the model to avoid warnings
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Load the face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Ensure the 'temp' directory exists
if not os.path.exists('temp'):
    os.makedirs('temp')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        try:
            # Save the file temporarily
            file_path = os.path.join('temp', file.filename)
            file.save(file_path)
            
            # Read the image using OpenCV
            img = cv2.imread(file_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) == 0:
                # Clean up
                os.remove(file_path)
                return jsonify({'result': 'Not a human face'}), 200
            
            # If a face is detected, proceed with fake/real detection
            img = image.load_img(file_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0
            
            # Make prediction
            prediction = model.predict(img_array)[0][0]
            
            # Clean up
            os.remove(file_path)
            
            result = 'Fake' if prediction > 0.5 else 'Real'
            confidence = float(prediction) if result == 'Fake' else float(1 - prediction)
            
            return jsonify({
                'result': result,
                'confidence': confidence * 100  # Convert to percentage
            }), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
