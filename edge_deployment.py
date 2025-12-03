"""
Edge AI Deployment Script for Raspberry Pi
=========================================

This script demonstrates how to deploy the TensorFlow Lite model on a Raspberry Pi
for real-time recyclable item classification using camera input.

Requirements:
- Raspberry Pi 4B (recommended)
- Camera module or USB camera
- TensorFlow Lite runtime
- OpenCV for image processing

Author: Joseph Majanja
Date: November 27, 2025
"""

import tensorflow as tf
import cv2
import numpy as np
import json
import time
import argparse
from pathlib import Path
import threading
import queue

class EdgeAIRecyclableDetector:
    """
    Real-time recyclable item detector for edge devices.
    """
    
    def __init__(self, model_path, config_path=None):
        """
        Initialize the edge AI detector.
        
        Args:
            model_path: Path to TensorFlow Lite model
            config_path: Path to configuration JSON file
        """
        self.model_path = model_path
        self.config = self._load_config(config_path)
        
        # Initialize TensorFlow Lite interpreter
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get model input/output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Model specifications
        self.input_shape = self.input_details[0]['shape'][1:4]  # Remove batch dimension
        self.class_names = self.config['class_names']
        
        # Performance monitoring
        self.inference_times = queue.Queue(maxsize=100)
        self.fps_counter = 0
        self.fps_start_time = time.time()
        
        print(f"Model loaded successfully!")
        print(f"Input shape: {self.input_shape}")
        print(f"Classes: {', '.join(self.class_names)}")
    
    def _load_config(self, config_path):
        """Load configuration from JSON file."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            # Default configuration
            return {
                'class_names': ['plastic', 'glass', 'metal', 'paper', 'cardboard', 'organic'],
                'confidence_threshold': 0.5,
                'input_size': (224, 224),
                'camera_resolution': (640, 480),
                'fps_target': 30
            }
    
    def preprocess_image(self, image):
        """
        Preprocess camera image for model input.
        
        Args:
            image: OpenCV image (BGR format)
            
        Returns:
            Preprocessed image ready for model input
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        image_resized = cv2.resize(image_rgb, self.config['input_size'])
        
        # Normalize to [0, 1] range
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        image_batch = np.expand_dims(image_normalized, axis=0)
        
        return image_batch
    
    def predict(self, processed_image):
        """
        Run inference on preprocessed image.
        
        Args:
            processed_image: Preprocessed image batch
            
        Returns:
            Dictionary with prediction results
        """
        start_time = time.time()
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], processed_image)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output
        output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        
        inference_time = time.time() - start_time
        
        # Update performance metrics
        if not self.inference_times.full():
            self.inference_times.put(inference_time)
        else:
            self.inference_times.get()
            self.inference_times.put(inference_time)
        
        # Get prediction details
        predicted_class_idx = np.argmax(output)
        confidence = float(output[predicted_class_idx])
        predicted_class = self.class_names[predicted_class_idx]
        
        return {
            'class_idx': predicted_class_idx,
            'class_name': predicted_class,
            'confidence': confidence,
            'inference_time': inference_time,
            'all_probabilities': output.tolist()
        }
    
    def draw_results(self, image, prediction):
        """
        Draw prediction results on image.
        
        Args:
            image: Original OpenCV image
            prediction: Prediction dictionary from predict()
            
        Returns:
            Image with overlaid results
        """
        height, width = image.shape[:2]
        
        # Only show results if confidence is above threshold
        if prediction['confidence'] < self.config['confidence_threshold']:
            text = "Low confidence"
            color = (128, 128, 128)  # Gray
        else:
            text = f"{prediction['class_name']}: {prediction['confidence']:.2f}"
            color = self._get_class_color(prediction['class_idx'])
        
        # Draw main prediction
        cv2.rectangle(image, (10, 10), (width-10, 80), (0, 0, 0), -1)
        cv2.putText(image, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        
        # Draw inference time
        time_text = f"Inference: {prediction['inference_time']*1000:.1f}ms"
        cv2.putText(image, time_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw FPS
        current_time = time.time()
        self.fps_counter += 1
        if current_time - self.fps_start_time >= 1.0:
            fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps = fps
            self.fps_counter = 0
            self.fps_start_time = current_time
        
        if hasattr(self, 'fps'):
            fps_text = f"FPS: {self.fps:.1f}"
            cv2.putText(image, fps_text, (width-150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Draw confidence bar for all classes
        if prediction['confidence'] >= self.config['confidence_threshold']:
            self._draw_confidence_bars(image, prediction['all_probabilities'])
        
        return image
    
    def _get_class_color(self, class_idx):
        """Get color for each class."""
        colors = [
            (255, 100, 100),  # plastic - light red
            (100, 255, 100),  # glass - light green
            (100, 100, 255),  # metal - light blue
            (255, 255, 100),  # paper - yellow
            (255, 100, 255),  # cardboard - magenta
            (100, 255, 255),  # organic - cyan
        ]
        return colors[class_idx % len(colors)]
    
    def _draw_confidence_bars(self, image, probabilities):
        """Draw confidence bars for all classes."""
        height, width = image.shape[:2]
        bar_width = 200
        bar_height = 15
        start_x = width - bar_width - 20
        start_y = 60
        
        for i, (class_name, prob) in enumerate(zip(self.class_names, probabilities)):
            y = start_y + i * (bar_height + 5)
            
            # Background bar
            cv2.rectangle(image, (start_x, y), (start_x + bar_width, y + bar_height), (50, 50, 50), -1)
            
            # Confidence bar
            confidence_width = int(bar_width * prob)
            color = self._get_class_color(i)
            cv2.rectangle(image, (start_x, y), (start_x + confidence_width, y + bar_height), color, -1)
            
            # Class name and percentage
            text = f"{class_name}: {prob:.2f}"
            cv2.putText(image, text, (start_x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def run_camera_detection(self, camera_id=0, save_results=False):
        """
        Run real-time detection using camera input.
        
        Args:
            camera_id: Camera device ID (0 for default camera)
            save_results: Whether to save detection results
        """
        print(f"Starting camera detection (Camera ID: {camera_id})")
        print("Press 'q' to quit, 's' to save current frame, 'p' to pause")
        
        # Initialize camera
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['camera_resolution'][0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['camera_resolution'][1])
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return
        
        # Results storage
        detection_results = []
        frame_count = 0
        paused = False
        
        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        print("Error: Could not read frame")
                        break
                    
                    # Preprocess and predict
                    processed_frame = self.preprocess_image(frame)
                    prediction = self.predict(processed_frame)
                    
                    # Store results if requested
                    if save_results:
                        detection_results.append({
                            'frame': frame_count,
                            'timestamp': time.time(),
                            'prediction': prediction
                        })
                    
                    # Draw results on frame
                    display_frame = self.draw_results(frame.copy(), prediction)
                    frame_count += 1
                else:
                    display_frame = frame  # Show last frame when paused
                
                # Display frame
                cv2.imshow('Edge AI - Recyclable Detection', display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s') and not paused:
                    # Save current frame and prediction
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"detection_{timestamp}.jpg"
                    cv2.imwrite(filename, display_frame)
                    print(f"Saved frame: {filename}")
                elif key == ord('p'):
                    paused = not paused
                    print(f"{'Paused' if paused else 'Resumed'}")
        
        except KeyboardInterrupt:
            print("\\nDetection interrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Save results if requested
            if save_results and detection_results:
                results_file = f"detection_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
                with open(results_file, 'w') as f:
                    json.dump(detection_results, f, indent=2)
                print(f"Detection results saved to: {results_file}")
            
            # Print performance summary
            self._print_performance_summary()
    
    def _print_performance_summary(self):
        """Print performance statistics."""
        if self.inference_times.empty():
            return
        
        times = []
        while not self.inference_times.empty():
            times.append(self.inference_times.get())
        
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        std_time = np.std(times)
        
        print(f"\\n=== Performance Summary ===")
        print(f"Average inference time: {avg_time*1000:.1f} ms")
        print(f"Min inference time: {min_time*1000:.1f} ms")
        print(f"Max inference time: {max_time*1000:.1f} ms")
        print(f"Std deviation: {std_time*1000:.1f} ms")
        print(f"Theoretical max FPS: {1/avg_time:.1f}")
    
    def benchmark_model(self, num_iterations=1000):
        """
        Benchmark model performance.
        
        Args:
            num_iterations: Number of benchmark iterations
        """
        print(f"Benchmarking model performance ({num_iterations} iterations)...")
        
        # Create dummy input
        dummy_input = np.random.random((1, *self.input_shape)).astype(np.float32)
        
        # Warm-up runs
        for _ in range(10):
            self.interpreter.set_tensor(self.input_details[0]['index'], dummy_input)
            self.interpreter.invoke()
        
        # Benchmark runs
        times = []
        for i in range(num_iterations):
            start_time = time.time()
            
            self.interpreter.set_tensor(self.input_details[0]['index'], dummy_input)
            self.interpreter.invoke()
            
            times.append(time.time() - start_time)
            
            if (i + 1) % 100 == 0:
                print(f"Completed {i + 1}/{num_iterations} iterations")
        
        # Calculate statistics
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        std_time = np.std(times)
        p95_time = np.percentile(times, 95)
        p99_time = np.percentile(times, 99)
        
        print(f"\\n=== Benchmark Results ===")
        print(f"Average inference time: {avg_time*1000:.2f} ms")
        print(f"Min inference time: {min_time*1000:.2f} ms")
        print(f"Max inference time: {max_time*1000:.2f} ms")
        print(f"95th percentile: {p95_time*1000:.2f} ms")
        print(f"99th percentile: {p99_time*1000:.2f} ms")
        print(f"Standard deviation: {std_time*1000:.2f} ms")
        print(f"Theoretical max FPS: {1/avg_time:.1f}")
        print(f"Practical max FPS (95% reliable): {1/p95_time:.1f}")


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description="Edge AI Recyclable Item Detector")
    parser.add_argument("--model", required=True, help="Path to TensorFlow Lite model")
    parser.add_argument("--config", help="Path to configuration JSON file")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--save-results", action="store_true", help="Save detection results")
    parser.add_argument("--iterations", type=int, default=1000, help="Benchmark iterations")
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        return
    
    try:
        # Initialize detector
        detector = EdgeAIRecyclableDetector(args.model, args.config)
        
        if args.benchmark:
            # Run benchmark
            detector.benchmark_model(args.iterations)
        else:
            # Run real-time detection
            detector.run_camera_detection(args.camera, args.save_results)
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()