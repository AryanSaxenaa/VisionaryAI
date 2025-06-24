import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json
import os

# --- Configuration: Define paths relative to this file ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '../models/eye_disease_model.h5')
CLASS_JSON_PATH = os.path.join(BASE_DIR, '../models/class_indices.json')
IMG_SIZE = 224

# --- Load Model and Class Names (once, on startup) ---
# This block uses tf.keras as intended with modern TensorFlow
try:
    # Use tf.keras.models.load_model
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_JSON_PATH) as f:
        class_names = json.load(f)
    print("✅ Model and class names loaded successfully using tf.keras.")
except Exception as e:
    print(f"❌ Error loading model or class names: {e}")
    model = None
    class_names = {}

def preprocess_image(image_bytes):
    """Prepares the uploaded image bytes for the model."""
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    # Use tf.keras.preprocessing.image.img_to_array
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    # Create a batch of 1
    img_array = np.expand_dims(img_array, axis=0)
    # Normalize the image
    return img_array / 255.0

def predict(image_bytes):
    """Makes a prediction on the preprocessed image."""
    if model is None:
        raise RuntimeError("Model is not loaded. Please check for errors on startup.")

    processed_image = preprocess_image(image_bytes)
    prediction = model.predict(processed_image)

    predicted_class_index = np.argmax(prediction[0])
    confidence = float(np.max(prediction[0]))

    full_class_name = class_names.get(str(predicted_class_index), "Unknown")
    display_name = full_class_name.split('.', 2)[-1].replace("'", "")

    return {"prediction": display_name, "confidence": confidence}

def get_model():
    """Returns the globally loaded model."""
    return model