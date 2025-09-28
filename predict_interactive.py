#!/usr/bin/env python3
"""
Interactive Plant Disease Prediction Script
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
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
    
    # Your dataset classes
    class_names = [
        'Pepper__bell___Bacterial_spot',
        'Pepper__bell___healthy', 
        'PlantVillage',
        'Potato___Early_blight',
        'Potato___Late_blight', 
        'Potato___healthy',
        'Tomato_Bacterial_spot',
        'Tomato_Early_blight',
        'Tomato_Late_blight',
        'Tomato_Leaf_Mold',
        'Tomato_Septoria_leaf_spot',
        'Tomato_Spider_mites_Two_spotted_spider_mite',
        'Tomato__Target_Spot',
        'Tomato__Tomato_YellowLeaf__Curl_Virus',
        'Tomato__Tomato_mosaic_virus',
        'Tomato_healthy'
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
    print("Plant Disease Detection - Interactive Mode")
    print("="*50)
    
    while True:
        # Ask for image path
        image_path = input("\nEnter the path to your plant image (or 'quit' to exit): ").strip()
        
        if image_path.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not image_path:
            print("Please enter a valid image path.")
            continue
        
        # Remove quotes if user added them
        image_path = image_path.strip('"').strip("'")
        
        load_model_and_predict(image_path)
        
        # Ask if user wants to test another image
        another = input("\nTest another image? (y/n): ").strip().lower()
        if another not in ['y', 'yes']:
            print("Goodbye!")
            break

if __name__ == "__main__":
    main()