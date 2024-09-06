from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)
CORS(app)  # Allow Cross-Origin requests

# Load model
model = load_model('fruit_detection_model.h5')

# Compile the model to avoid warnings
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

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
        # Save the file temporarily
        file_path = os.path.join('temp', file.filename)
        file.save(file_path)
        
        # Preprocess the image
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)  # Ensure the app runs on the correct port
