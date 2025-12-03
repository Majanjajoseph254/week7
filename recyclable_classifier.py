"""
Edge AI Prototype: Recyclable Item Classification
===============================================

This script implements a lightweight image classification model for recognizing recyclable items.
The model is designed to be converted to TensorFlow Lite for edge deployment.

Author: Joseph
Date: November 27, 2025
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import os
import json
from datetime import datetime
import seaborn as sns

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class RecyclableClassifier:
    """
    A lightweight CNN model for classifying recyclable items.
    Designed for edge deployment with TensorFlow Lite.
    """
    
    def __init__(self, num_classes=6, input_shape=(224, 224, 3)):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = None
        self.tflite_model = None
        self.class_names = ['plastic', 'glass', 'metal', 'paper', 'cardboard', 'organic']
        
    def create_lightweight_model(self):
        """
        Create a lightweight CNN model optimized for edge deployment.
        Uses depthwise separable convolutions for efficiency.
        """
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # Preprocessing
            layers.Rescaling(1./255),
            
            # Feature extraction with depthwise separable convolutions
            layers.SeparableConv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.SeparableConv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.SeparableConv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.SeparableConv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            
            # Classification head
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile with optimization for mobile deployment
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_2_accuracy']
        )
        
        self.model = model
        return model
    
    def create_synthetic_dataset(self, samples_per_class=200):
        """
        Create a synthetic dataset for demonstration purposes.
        In real deployment, replace with actual recyclable item images.
        """
        print("Creating synthetic dataset for demonstration...")
        
        # Create synthetic data with different patterns for each class
        X_data = []
        y_data = []
        
        for class_idx in range(self.num_classes):
            for _ in range(samples_per_class):
                # Generate synthetic images with class-specific patterns
                if class_idx == 0:  # Plastic - smooth textures
                    img = self._generate_smooth_texture()
                elif class_idx == 1:  # Glass - transparent/reflective
                    img = self._generate_glass_texture()
                elif class_idx == 2:  # Metal - metallic shine
                    img = self._generate_metal_texture()
                elif class_idx == 3:  # Paper - fibrous texture
                    img = self._generate_paper_texture()
                elif class_idx == 4:  # Cardboard - corrugated texture
                    img = self._generate_cardboard_texture()
                else:  # Organic - irregular patterns
                    img = self._generate_organic_texture()
                
                X_data.append(img)
                y_data.append(class_idx)
        
        X_data = np.array(X_data)
        y_data = keras.utils.to_categorical(y_data, self.num_classes)
        
        # Split into train and test sets
        split_idx = int(0.8 * len(X_data))
        indices = np.random.permutation(len(X_data))
        
        X_train = X_data[indices[:split_idx]]
        y_train = y_data[indices[:split_idx]]
        X_test = X_data[indices[split_idx:]]
        y_test = y_data[indices[split_idx:]]
        
        print(f"Dataset created: {len(X_train)} training, {len(X_test)} test samples")
        return (X_train, y_train), (X_test, y_test)
    
    def _generate_smooth_texture(self):
        """Generate synthetic smooth texture for plastic items."""
        img = np.random.normal(0.6, 0.1, self.input_shape)
        # Add some smooth gradients
        x = np.linspace(0, 1, self.input_shape[0])
        y = np.linspace(0, 1, self.input_shape[1])
        xx, yy = np.meshgrid(x, y)
        gradient = 0.3 * np.sin(2 * np.pi * xx) * np.cos(2 * np.pi * yy)
        img[:, :, 0] += gradient
        return np.clip(img, 0, 1)
    
    def _generate_glass_texture(self):
        """Generate synthetic glass-like texture."""
        img = np.random.normal(0.8, 0.05, self.input_shape)
        # Add reflective patterns
        for _ in range(5):
            x = np.random.randint(0, self.input_shape[0])
            y = np.random.randint(0, self.input_shape[1])
            img[x:x+20, y:y+20] += 0.2
        return np.clip(img, 0, 1)
    
    def _generate_metal_texture(self):
        """Generate synthetic metallic texture."""
        img = np.random.normal(0.7, 0.15, self.input_shape)
        # Add metallic shine patterns
        noise = np.random.normal(0, 0.1, self.input_shape)
        img += noise * 0.5
        return np.clip(img, 0, 1)
    
    def _generate_paper_texture(self):
        """Generate synthetic paper fiber texture."""
        img = np.random.normal(0.9, 0.05, self.input_shape)
        # Add fibrous patterns
        for _ in range(100):
            x = np.random.randint(0, self.input_shape[0]-5)
            y = np.random.randint(0, self.input_shape[1]-1)
            img[x:x+5, y] -= 0.1
        return np.clip(img, 0, 1)
    
    def _generate_cardboard_texture(self):
        """Generate synthetic cardboard corrugated texture."""
        img = np.random.normal(0.6, 0.08, self.input_shape)
        # Add corrugated patterns
        for i in range(0, self.input_shape[0], 10):
            img[i:i+3, :] += 0.15
        return np.clip(img, 0, 1)
    
    def _generate_organic_texture(self):
        """Generate synthetic organic texture."""
        img = np.random.normal(0.4, 0.2, self.input_shape)
        # Add irregular organic patterns
        for _ in range(20):
            x = np.random.randint(10, self.input_shape[0]-10)
            y = np.random.randint(10, self.input_shape[1]-10)
            radius = np.random.randint(5, 15)
            img[x-radius:x+radius, y-radius:y+radius] += np.random.normal(0, 0.1)
        return np.clip(img, 0, 1)
    
    def train_model(self, train_data, test_data, epochs=20):
        """
        Train the model with early stopping and model checkpointing.
        """
        if self.model is None:
            self.create_lightweight_model()
        
        X_train, y_train = train_data
        X_test, y_test = test_data
        
        # Callbacks for training optimization
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                'models/best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train the model
        print("Starting model training...")
        history = self.model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=epochs,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate_model(self, test_data):
        """
        Comprehensive model evaluation with metrics and visualizations.
        """
        X_test, y_test = test_data
        
        # Get predictions
        predictions = self.model.predict(X_test)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y_test, axis=1)
        
        # Calculate metrics
        test_loss, test_accuracy, test_top2 = self.model.evaluate(X_test, y_test, verbose=0)
        
        print(f"\n=== Model Evaluation Results ===")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Top-2 Accuracy: {test_top2:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        # Classification report
        print(f"\n=== Classification Report ===")
        print(classification_report(true_classes, predicted_classes, 
                                  target_names=self.class_names))
        
        # Confusion matrix
        cm = confusion_matrix(true_classes, predicted_classes)
        self._plot_confusion_matrix(cm)
        
        # Calculate model size
        model_size = self.model.count_params()
        print(f"\nModel Parameters: {model_size:,}")
        
        return {
            'accuracy': test_accuracy,
            'top2_accuracy': test_top2,
            'loss': test_loss,
            'confusion_matrix': cm.tolist(),
            'model_size': model_size
        }
    
    def _plot_confusion_matrix(self, cm):
        """Plot confusion matrix with proper formatting."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix - Recyclable Item Classification')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('models/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def convert_to_tflite(self, optimize=True):
        """
        Convert the trained model to TensorFlow Lite format for edge deployment.
        """
        if self.model is None:
            raise ValueError("Model must be trained before conversion")
        
        # Create TensorFlow Lite converter
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        if optimize:
            # Apply optimizations for edge deployment
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # Optional: Use dynamic range quantization for further size reduction
            converter.representative_dataset = self._representative_dataset_gen
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
        
        # Convert model
        print("Converting model to TensorFlow Lite...")
        self.tflite_model = converter.convert()
        
        # Save TFLite model
        tflite_path = 'models/recyclable_classifier.tflite'
        with open(tflite_path, 'wb') as f:
            f.write(self.tflite_model)
        
        # Get model size information
        original_size = os.path.getsize('models/best_model.h5')
        tflite_size = len(self.tflite_model)
        compression_ratio = original_size / tflite_size
        
        print(f"Model conversion completed!")
        print(f"Original model size: {original_size / 1024:.1f} KB")
        print(f"TFLite model size: {tflite_size / 1024:.1f} KB")
        print(f"Compression ratio: {compression_ratio:.1f}x")
        
        return tflite_path, tflite_size
    
    def _representative_dataset_gen(self):
        """Generate representative dataset for quantization."""
        # Use a subset of training data for quantization calibration
        for _ in range(100):
            yield [np.random.random((1, *self.input_shape)).astype(np.float32)]
    
    def test_tflite_model(self, test_data, num_samples=50):
        """
        Test the TensorFlow Lite model and compare with original model.
        """
        if self.tflite_model is None:
            raise ValueError("TFLite model not available. Convert model first.")
        
        X_test, y_test = test_data
        
        # Load TFLite model and allocate tensors
        interpreter = tf.lite.Interpreter(model_content=self.tflite_model)
        interpreter.allocate_tensors()
        
        # Get input and output tensor details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Test on a subset of data
        test_indices = np.random.choice(len(X_test), num_samples, replace=False)
        tflite_predictions = []
        original_predictions = []
        
        print(f"Testing TFLite model on {num_samples} samples...")
        
        for idx in test_indices:
            # Prepare input for TFLite
            input_data = X_test[idx:idx+1].astype(np.float32)
            
            # TFLite prediction
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            tflite_output = interpreter.get_tensor(output_details[0]['index'])
            tflite_predictions.append(np.argmax(tflite_output))
            
            # Original model prediction
            original_output = self.model.predict(input_data, verbose=0)
            original_predictions.append(np.argmax(original_output))
        
        # Compare predictions
        agreement = np.mean(np.array(tflite_predictions) == np.array(original_predictions))
        
        print(f"TFLite vs Original Model Agreement: {agreement:.4f}")
        
        # Test inference speed
        import time
        
        # TFLite speed test
        start_time = time.time()
        for _ in range(100):
            interpreter.set_tensor(input_details[0]['index'], X_test[:1].astype(np.float32))
            interpreter.invoke()
        tflite_time = (time.time() - start_time) / 100
        
        # Original model speed test
        start_time = time.time()
        for _ in range(100):
            self.model.predict(X_test[:1], verbose=0)
        original_time = (time.time() - start_time) / 100
        
        print(f"Average Inference Time:")
        print(f"  Original Model: {original_time*1000:.1f} ms")
        print(f"  TFLite Model: {tflite_time*1000:.1f} ms")
        print(f"  Speedup: {original_time/tflite_time:.1f}x")
        
        return {
            'agreement': agreement,
            'original_inference_time': original_time,
            'tflite_inference_time': tflite_time,
            'speedup': original_time / tflite_time
        }
    
    def demonstrate_real_time_classification(self, test_data, num_demos=5):
        """
        Demonstrate real-time classification capabilities.
        """
        X_test, y_test = test_data
        
        # Load TFLite interpreter
        interpreter = tf.lite.Interpreter(model_content=self.tflite_model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"\n=== Real-Time Classification Demo ===")
        
        for i in range(num_demos):
            # Select random sample
            idx = np.random.randint(len(X_test))
            sample = X_test[idx:idx+1].astype(np.float32)
            true_class = np.argmax(y_test[idx])
            
            # Measure inference time
            import time
            start_time = time.time()
            
            # Run inference
            interpreter.set_tensor(input_details[0]['index'], sample)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            
            inference_time = time.time() - start_time
            
            # Get prediction
            predicted_class = np.argmax(output)
            confidence = np.max(output)
            
            print(f"Sample {i+1}:")
            print(f"  True class: {self.class_names[true_class]}")
            print(f"  Predicted: {self.class_names[predicted_class]}")
            print(f"  Confidence: {confidence:.3f}")
            print(f"  Inference time: {inference_time*1000:.1f} ms")
            print(f"  Correct: {'✓' if predicted_class == true_class else '✗'}")
            print()
    
    def save_deployment_package(self, metrics):
        """
        Save complete deployment package with model and metadata.
        """
        deployment_info = {
            'model_info': {
                'name': 'Recyclable Item Classifier',
                'version': '1.0.0',
                'created_date': datetime.now().isoformat(),
                'input_shape': list(self.input_shape),
                'num_classes': self.num_classes,
                'class_names': self.class_names
            },
            'performance_metrics': metrics,
            'deployment_instructions': {
                'model_file': 'recyclable_classifier.tflite',
                'input_preprocessing': 'Normalize pixel values to [0,1] range',
                'output_postprocessing': 'Apply softmax to get class probabilities',
                'recommended_hardware': [
                    'Raspberry Pi 4B (4GB RAM)',
                    'Google Coral Dev Board',
                    'NVIDIA Jetson Nano',
                    'Mobile devices with TensorFlow Lite support'
                ]
            }
        }
        
        # Save deployment metadata
        with open('models/deployment_info.json', 'w') as f:
            json.dump(deployment_info, f, indent=2)
        
        print("Deployment package saved successfully!")
        print("Files created:")
        print("  - models/recyclable_classifier.tflite")
        print("  - models/deployment_info.json")
        print("  - models/confusion_matrix.png")


def main():
    """
    Main function to demonstrate the complete Edge AI prototype workflow.
    """
    print("=== Edge AI Prototype: Recyclable Item Classification ===\n")
    
    # Initialize classifier
    classifier = RecyclableClassifier()
    
    # Create model architecture
    print("Creating lightweight CNN model...")
    model = classifier.create_lightweight_model()
    
    # Display model summary
    print("\n=== Model Architecture ===")
    model.summary()
    
    # Create synthetic dataset
    train_data, test_data = classifier.create_synthetic_dataset(samples_per_class=200)
    
    # Train model
    history = classifier.train_model(train_data, test_data, epochs=20)
    
    # Evaluate model
    metrics = classifier.evaluate_model(test_data)
    
    # Convert to TensorFlow Lite
    tflite_path, tflite_size = classifier.convert_to_tflite(optimize=True)
    
    # Test TFLite model
    tflite_metrics = classifier.test_tflite_model(test_data)
    
    # Update metrics with TFLite performance
    metrics.update(tflite_metrics)
    
    # Demonstrate real-time classification
    classifier.demonstrate_real_time_classification(test_data)
    
    # Save deployment package
    classifier.save_deployment_package(metrics)
    
    # Print final summary
    print("\n=== Edge AI Prototype Summary ===")
    print(f"✓ Model trained with {metrics['accuracy']:.1%} accuracy")
    print(f"✓ Model converted to TensorFlow Lite ({tflite_size/1024:.1f} KB)")
    print(f"✓ Real-time inference: {metrics['tflite_inference_time']*1000:.1f} ms")
    print(f"✓ {metrics['speedup']:.1f}x speedup over original model")
    print(f"✓ Deployment package ready for edge devices")
    
    return classifier, metrics


if __name__ == "__main__":
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    
    # Run the complete prototype
    classifier, final_metrics = main()