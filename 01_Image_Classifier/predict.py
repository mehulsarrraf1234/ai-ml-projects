"""
Prediction Script for Image Classification Model
"""

import numpy as np
import argparse
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import cv2
import os

# CIFAR-10 class names
CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def load_and_preprocess_image(img_path, target_size=(32, 32)):
    """
    Load and preprocess an image for prediction.
    
    Args:
        img_path: Path to the image file
        target_size: Target size for the image
    
    Returns:
        Preprocessed image array
    """
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
    
    # Load image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize
    img = cv2.resize(img, target_size)
    
    # Normalize
    img = img.astype('float32') / 255.0
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img


def predict_image(model, img_path, class_names=CLASS_NAMES):
    """
    Make prediction on a single image.
    
    Args:
        model: Trained Keras model
        img_path: Path to the image
        class_names: List of class names
    
    Returns:
        Predicted class, confidence score, all probabilities
    """
    # Load and preprocess
    img = load_and_preprocess_image(img_path)
    
    # Make prediction
    predictions = model.predict(img, verbose=0)[0]
    
    # Get results
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    
    return predicted_class, confidence, predictions


def predict_and_display(model, img_path, class_names=CLASS_NAMES):
    """
    Make prediction and display results.
    
    Args:
        model: Trained Keras model
        img_path: Path to the image
        class_names: List of class names
    """
    print(f"\n🔍 Analyzing image: {img_path}")
    print("-" * 40)
    
    # Get prediction
    predicted_class, confidence, predictions = predict_image(model, img_path, class_names)
    
    # Display results
    print(f"🎯 Predicted Class: {predicted_class.upper()}")
    print(f"📊 Confidence: {confidence:.2f}%")
    
    # Show top 3 predictions
    print("\n📈 Top 3 Predictions:")
    top_3_idx = np.argsort(predictions)[-3:][::-1]
    for i, idx in enumerate(top_3_idx, 1):
        print(f"   {i}. {class_names[idx]}: {predictions[idx]*100:.2f}%")
    
    # Display image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.title(f'Predicted: {predicted_class.upper()} ({confidence:.2f}%)', fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('prediction_result.png', dpi=150)
    plt.show()
    print("\n💾 Result saved to 'prediction_result.png'")


def batch_predict(model, image_dir, class_names=CLASS_NAMES):
    """
    Make predictions on a batch of images.
    
    Args:
        model: Trained Keras model
        image_dir: Directory containing images
        class_names: List of class names
    """
    if not os.path.exists(image_dir):
        print(f"Directory not found: {image_dir}")
        return
    
    image_files = [f for f in os.listdir(image_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"No images found in {image_dir}")
        return
    
    print(f"\n📁 Processing {len(image_files)} images...")
    print("-" * 50)
    
    results = []
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        pred_class, conf, _ = predict_image(model, img_path, class_names)
        results.append((img_file, pred_class, conf))
        print(f"  {img_file}: {pred_class} ({conf:.2f}%)")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Classification Prediction')
    parser.add_argument('--image', '-i', type=str, required=True,
                        help='Path to the image file')
    parser.add_argument('--model', '-m', type=str, default='image_classifier_model.h5',
                        help='Path to the trained model')
    parser.add_argument('--batch', '-b', type=str, default=None,
                        help='Directory for batch prediction')
    
    args = parser.parse_args()
    
    # Load model
    print("📦 Loading model...")
    try:
        model = load_model(args.model)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Note: Please train the model first using 'python main.py'")
        exit(1)
    
    # Make prediction
    if args.batch:
        batch_predict(model, args.batch, CLASS_NAMES)
    else:
        predict_and_display(model, args.image, CLASS_NAMES)
