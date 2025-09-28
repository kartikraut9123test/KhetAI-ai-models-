#!/usr/bin/env python3
"""
Plant Disease Prediction Script
Usage: python predict_disease.py <image_path>
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import sys
import os

def load_model_and_predict(image_path, model_path='plant_disease_model.h5'):
    """Load trained model and predict disease from image"""
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' not found!")
        print("Please train the model first by running PlantDiseaseDetection.py")
        return
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Image file '{image_path}' not found!")
        return
    
    # Load model
    print("Loading model...")
    model = tf.keras.models.load_model(model_path)
    
    # Common plant disease classes (update based on your dataset)
    class_names = [
        'Apple___Apple_scab',
        'Apple___Black_rot',
        'Apple___Cedar_apple_rust',
        'Apple___healthy',
        'Blueberry___healthy',
        'Cherry_(including_sour)___Powdery_mildew',
        'Cherry_(including_sour)___healthy',
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
        'Corn_(maize)___Common_rust_',
        'Corn_(maize)___Northern_Leaf_Blight',
        'Corn_(maize)___healthy',
        'Grape___Black_rot',
        'Grape___Esca_(Black_Measles)',
        'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
        'Grape___healthy',
        'Orange___Haunglongbing_(Citrus_greening)',
        'Peach___Bacterial_spot',
        'Peach___healthy',
        'Pepper,_bell___Bacterial_spot',
        'Pepper,_bell___healthy',
        'Potato___Early_blight',
        'Potato___Late_blight',
        'Potato___healthy',
        'Raspberry___healthy',
        'Soybean___healthy',
        'Squash___Powdery_mildew',
        'Strawberry___Leaf_scorch',
        'Strawberry___healthy',
        'Tomato___Bacterial_spot',
        'Tomato___Early_blight',
        'Tomato___Late_blight',
        'Tomato___Leaf_Mold',
        'Tomato___Septoria_leaf_spot',
        'Tomato___Spider_mites Two-spotted_spider_mite',
        'Tomato___Target_Spot',
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
        'Tomato___Tomato_mosaic_virus',
        'Tomato___healthy'
    ]
    
    try:
        # Load and preprocess image
        print(f"Processing image: {image_path}")
        img = load_img(image_path, target_size=(128, 128))
        img_array = img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        
        # Make prediction
        print("Making prediction...")
        predictions = model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        # Get class name
        if predicted_class_idx < len(class_names):
            predicted_class = class_names[predicted_class_idx]
        else:
            predicted_class = f"Class_{predicted_class_idx}"
        
        # Display results
        print("\n" + "="*50)
        print("PREDICTION RESULTS")
        print("="*50)
        print(f"Image: {os.path.basename(image_path)}")
        print(f"Predicted Disease: {predicted_class}")
        print(f"Confidence: {confidence:.2%}")
        print("="*50)
        
        # Show top 3 predictions
        top_3_idx = np.argsort(predictions[0])[-3:][::-1]
        print("\nTop 3 Predictions:")
        for i, idx in enumerate(top_3_idx, 1):
            class_name = class_names[idx] if idx < len(class_names) else f"Class_{idx}"
            conf = predictions[0][idx]
            print(f"{i}. {class_name}: {conf:.2%}")
        
    except Exception as e:
        print(f"Error processing image: {e}")

def main():
    # Option 1: Use command line argument
    if len(sys.argv) == 2:
        image_path = sys.argv[1]
    else:
        # Option 2: Set your image path directly here
        image_path = "test_image.jpg"  # Change this to your image path
        print(f"No command line argument provided. Using default: {image_path}")
        print("Usage: python predict_disease.py <image_path>")
        print("Or modify the 'image_path' variable in the script")
    
    load_model_and_predict(image_path)

if __name__ == "__main__":
    main()