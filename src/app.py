
from flask import Flask, render_template, request
from predict import predict_image
import os

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    predicted_label, confidence = predict_image(file_path)
    return f"Predicted Class: {predicted_label}, Confidence: {confidence*100:.2f}%"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
