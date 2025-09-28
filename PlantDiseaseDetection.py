# Import Libraries
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Configure GPU memory growth to avoid OOM errors
def configure_gpu():
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU memory growth enabled")
        else:
            print("No GPU devices found. Running on CPU")
    except Exception as e:
        print(f"Error configuring GPU: {e}")
        print("Running on CPU")

# Set parameters - Reduced for memory efficiency
IMG_SIZE = 128  # Reduced from 224 to save memory
BATCH_SIZE = 16  # Reduced from 32 to save memory
EPOCHS = 5  # Reduced for faster training

# Dataset path - using os.path for cross-platform compatibility
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "archive (1)", "PlantVillage")

def create_model(num_classes):
    """Create an improved CNN model for plant disease detection"""
    model = models.Sequential([
        layers.Rescaling(1./255, input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        
        # First conv block
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        
        # Second conv block
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        
        # Third conv block
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def load_and_prepare_data():
    """Load and prepare training and validation datasets with data augmentation"""
    if not os.path.exists(data_dir):
        print(f"Data directory '{data_dir}' not found!")
        return None, None, None
    
    # Data augmentation for training
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomBrightness(0.2),
    ])
    
    # Load datasets
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE
    )
    
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE
    )
    
    # Extract class names before optimization
    class_names = train_ds.class_names
    
    # Apply data augmentation only to training dataset
    train_ds = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Optimize performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    return train_ds, val_ds, class_names

def predict_disease(model, image_path, class_names):
    """Predict disease from a single image"""
    try:
        # Load and preprocess image
        img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create batch
        
        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))
        
        return predicted_class, confidence
    except Exception as e:
        print(f"Error predicting image: {e}")
        return None, None

def plot_training_history(history):
    """Plot training and validation accuracy/loss"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def main():
    """Main training and evaluation function"""
    # Configure GPU
    configure_gpu()
    
    print("Loading data...")
    train_ds, val_ds, class_names = load_and_prepare_data()
    
    if train_ds is None:
        return None, None
    
    num_classes = len(class_names)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {class_names}")
    
    # Create and compile model
    model = create_model(num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top3_accuracy')]
    )
    
    # Display model summary
    model.summary()
    
    # Train model
    print("Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        verbose=1
    )
    
    # Evaluate model
    test_loss, test_acc = model.evaluate(val_ds, verbose=0)
    print(f'Validation accuracy: {test_acc:.4f}')
    
    # Save model
    model.save('plant_disease_model.h5')
    print("Model saved as 'plant_disease_model.h5'")
    
    # Plot training history
    plot_training_history(history)
    
    return model, class_names

if __name__ == "__main__":
    model, class_names = main()