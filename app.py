from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import json
import os
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Load trained model
model = load_model("MINI_FINAL.keras")

# Recycling methods dictionary
recycling_methods = {
    "battery": "Extract metals & chemicals, recycle at battery processing centers.",
    "biological": "Composting to produce organic fertilizers.",
    "cardboard": "Shredding & pulping to make new paper products.",
    "clothes": "Fiber recovery for new textiles or insulation materials.",
    "glass": "Melting & remolding into new glass products.",
    "metal": "Melting & reshaping into new metal items.",
    "paper": "Pulping & remanufacturing into recycled paper products.",
    "plastic": "Shredding, melting & remolding into new plastic products.",
    "shoes": "Grinding materials for use in new footwear or rubber products.",
    "trash": "Incineration for energy recovery or landfill disposal."
}

# Class labels
class_labels = list(recycling_methods.keys())

# Prediction function
def predict_waste(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array_expanded = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array_expanded)
    predicted_index = np.argmax(prediction)
    confidence = float(np.max(prediction))

    predicted_class = class_labels[predicted_index]
    method = recycling_methods.get(predicted_class, "No recycling method available.")

    return {
        "classified_waste": predicted_class,
        "confidence": round(confidence, 2),
        "recycling_method": method
    }

# Route: Home
@app.route('/')
def home():
    return "♻️ Waste Classification API is running!"

# Route: Prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Empty filename."}), 400

    filename = secure_filename(file.filename)
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, filename)
    file.save(file_path)

    result = predict_waste(file_path)
    os.remove(file_path)

    return jsonify(result)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
