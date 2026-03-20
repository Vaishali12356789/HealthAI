from flask import Flask, request, jsonify
import numpy as np
import cv2
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model('model.h5')

@app.route('/')
def home():
    return "HealthAI API"

@app.route('/predict', methods=['POST'])
def predict():
    img = request.files['image']
    img = cv2.imdecode(np.frombuffer(img.read(), np.uint8), cv2.IMREAD_COLOR)
    # Process image and predict
    return jsonify({'result': 'prediction'})

if __name__ == '__main__':
    app.run(debug=True)
