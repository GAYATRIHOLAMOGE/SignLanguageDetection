import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split

# Optimize TensorFlow for CPU
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)

print("TensorFlow version:", tf.__version__)
print("Running on:", "GPU" if tf.config.list_physical_devices('GPU') else "CPU")

# ==================== DATA PREPARATION ====================

def load_and_preprocess_data(train_dir, test_dir, img_size=(64, 64), quick_test=False):
    """
    Load ASL Alphabet dataset from Kaggle
    Set quick_test=True for faster testing (uses subset of data)
    """
    # Training data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.15,
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.15,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    # Validation data (no augmentation except rescaling)
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.15
    )
    
    batch_size = 64 if not quick_test else 32  # Larger batch for CPU efficiency
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    return train_generator, val_generator

# ==================== MODEL ARCHITECTURE ====================

def create_cnn_model(input_shape=(64, 64, 3), num_classes=29):
    """
    Create a CPU-optimized CNN model for ASL Alphabet detection
    Lighter architecture for faster training on CPU
    """
    model = keras.Sequential([
        # First Convolutional Block
        layers.Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fourth Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Flatten and Dense Layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def create_transfer_learning_model(input_shape=(64, 64, 3), num_classes=29):
    """
    Create a transfer learning model using MobileNetV2
    MUCH FASTER on CPU! Recommended for CPU training.
    """
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model layers for faster training
    base_model.trainable = False
    
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# ==================== TRAINING ====================

def train_model(model, train_gen, val_gen, epochs=50):
    """
    Compile and train the model
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7
    )
    
    checkpoint = keras.callbacks.ModelCheckpoint(
        'best_sign_language_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=[early_stopping, reduce_lr, checkpoint]
    )
    
    return history

# ==================== VISUALIZATION ====================

def plot_training_history(history):
    """
    Plot training and validation accuracy/loss
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Loss plot
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

# ==================== REAL-TIME DETECTION ====================

class SignLanguageDetector:
    def __init__(self, model_path, class_labels):
        self.model = keras.models.load_model(model_path)
        self.class_labels = class_labels
        self.img_size = (64, 64)
    
    def preprocess_frame(self, frame):
        """Preprocess frame for prediction"""
        img = cv2.resize(frame, self.img_size)
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    
    def predict(self, frame):
        """Predict ASL sign"""
        preprocessed = self.preprocess_frame(frame)
        prediction = self.model.predict(preprocessed, verbose=0)
        class_idx = np.argmax(prediction)
        confidence = prediction[0][class_idx]
        return self.class_labels[class_idx], confidence
    
    def detect_realtime(self):
        """Real-time detection using webcam"""
        cap = cv2.VideoCapture(0)
        
        # For smoothing predictions
        prediction_history = []
        history_size = 5
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Define ROI (Region of Interest)
            h, w, _ = frame.shape
            roi_size = 300
            roi_x = (w - roi_size) // 2
            roi_y = (h - roi_size) // 2
            
            roi = frame[roi_y:roi_y+roi_size, roi_x:roi_x+roi_size]
            
            # Get prediction
            label, confidence = self.predict(roi)
            
            # Add to history for smoothing
            prediction_history.append(label)
            if len(prediction_history) > history_size:
                prediction_history.pop(0)
            
            # Use most common prediction
            if len(prediction_history) == history_size:
                from collections import Counter
                label = Counter(prediction_history).most_common(1)[0][0]
            
            # Draw ROI rectangle
            cv2.rectangle(frame, (roi_x, roi_y), 
                         (roi_x+roi_size, roi_y+roi_size), (0, 255, 0), 3)
            
            # Create a semi-transparent overlay for text background
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (400, 100), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
            
            # Display prediction
            text = f"Sign: {label}"
            conf_text = f"Confidence: {confidence:.2%}"
            cv2.putText(frame, text, (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            cv2.putText(frame, conf_text, (20, 85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Instructions
            cv2.putText(frame, "Press 'q' to quit", (10, h-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow('ASL Alphabet Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    # ==================== CONFIGURATION ====================
    
    # Set paths (update these to match your folder structure)
   #BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    #TRAIN_DIR = os.path.join(BASE_DIR, 'asl_alphabet_train', 'asl_alphabet_train')
    #TEST_DIR = os.path.join(BASE_DIR, 'asl_alphabet_test', 'asl_alphabet_test')
    
    # Or use relative paths:
    TRAIN_DIR = 'asl_alphabet_train/asl_alphabet_train'
    TEST_DIR = 'asl_alphabet_test/asl_alphabet_test'
    
    # Model configuration
    IMG_SIZE = (64, 64)  # Smaller for faster CPU trainingS
    NUM_CLASSES = 29
    EPOCHS = 20  # Reduced for CPU
    USE_TRANSFER_LEARNING = True  # Set to True for FASTER training!
    QUICK_TEST = False  # Set to True for quick testing with less data
    
    print("="*60)
    print("ASL ALPHABET DETECTION - CPU OPTIMIZED")
    print("="*60)
    print(f"Image Size: {IMG_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Transfer Learning: {USE_TRANSFER_LEARNING}")
    print(f"Quick Test Mode: {QUICK_TEST}")
    print("="*60)
    
    # Verify dataset exists
    if not os.path.exists(TRAIN_DIR):
        print(f"\n❌ ERROR: Training directory not found!")
        print(f"Looking for: {TRAIN_DIR}")
        print("\nPlease:")
        print("1. Download the dataset from Kaggle")
        print("2. Extract it to the project folder")
        print("3. Update TRAIN_DIR path in the code")
        exit(1)
    
    # Step 1: Load data
    print("\n📁 Loading ASL Alphabet dataset...")
    train_gen, val_gen = load_and_preprocess_data(TRAIN_DIR, TEST_DIR, IMG_SIZE, QUICK_TEST)
    
    # Get class labels
    class_labels = list(train_gen.class_indices.keys())
    print(f"✓ Number of classes: {len(class_labels)}")
    print(f"✓ Classes: {sorted(class_labels)}")
    print(f"✓ Training samples: {train_gen.samples}")
    print(f"✓ Validation samples: {val_gen.samples}")
    
    # Step 2: Create model
    print("\n🧠 Creating model...")
    if USE_TRANSFER_LEARNING:
        print("Using MobileNetV2 (Transfer Learning) - Recommended for CPU!")
        model = create_transfer_learning_model(
            input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
            num_classes=NUM_CLASSES
        )
    else:
        print("Using Custom CNN - May take longer on CPU")
        model = create_cnn_model(
            input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
            num_classes=NUM_CLASSES
        )
    
    model.summary()
    
    # Step 3: Train model
    print("\n🚀 Training model...")
    print("⏱️  Estimated time on CPU:")
    if USE_TRANSFER_LEARNING:
        print("   - Transfer Learning: ~30-60 minutes")
    else:
        print("   - Custom CNN: ~2-4 hours")
    print("\n💡 Tip: Let it run and check back later!")
    print("="*60)
    
    import time
    start_time = time.time()
    
    history = train_model(model, train_gen, val_gen, epochs=EPOCHS)
    
    training_time = (time.time() - start_time) / 60
    print(f"\n✓ Training completed in {training_time:.1f} minutes!")
    
    # Step 4: Visualize results
    print("\n📊 Generating training history plots...")
    plot_training_history(history)
    
    # Step 5: Save model
    model.save('asl_alphabet_model.h5')
    print("\n💾 Model saved as 'asl_alphabet_model.h5'")
    
    # Step 6: Evaluate on validation set
    print("\n📈 Evaluating model on validation set...")
    val_loss, val_accuracy = model.evaluate(val_gen)
    print(f"✓ Validation Loss: {val_loss:.4f}")
    print(f"✓ Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
    
    # Step 7: Real-time detection (optional)
    print("\n" + "="*60)
    print("REAL-TIME DETECTION")
    print("="*60)
    response = input("Start real-time webcam detection? (y/n): ")
    if response.lower() == 'y':
        print("\n📹 Starting real-time detection...")
        print("Instructions:")
        print("  1. Position your hand in the green box")
        print("  2. Make ASL signs")
        print("  3. Press 'q' to quit")
        print("\nStarting in 3 seconds...")
        time.sleep(3)
        
        detector = SignLanguageDetector('best_sign_language_model.h5', class_labels)
        detector.detect_realtime()
    else:
        print("\n✓ Training complete! You can run real-time detection later.")
        print("\nTo use the model later, run:")
        print("  detector = SignLanguageDetector('best_sign_language_model.h5', class_labels)")
        print("  detector.detect_realtime()")
    
    print("\n" + "="*60)
    print("ALL DONE! 🎉")
    print("="*60)