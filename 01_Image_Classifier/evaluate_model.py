"""
Model Evaluation Script
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)
from tensorflow.keras.models import load_model
from data_loader import load_cifar10, create_tf_datasets

# CIFAR-10 class names
CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def evaluate_model(model_path='image_classifier_model.h5'):
    """
    Comprehensive model evaluation.
    
    Args:
        model_path: Path to the trained model
    """
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    # Load model
    print("\n📦 Loading model...")
    try:
        model = load_model(model_path)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Note: Please train the model first using 'python main.py'")
        return
    
    # Load test data
    print("\n📂 Loading test data...")
    (_, y_train), (x_test, y_test) = load_cifar10()
    print(f"Test samples: {len(x_test)}")
    
    # Get predictions
    print("\n🔮 Making predictions...")
    predictions = model.predict(x_test, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print("\n" + "=" * 40)
    print("📊 PERFORMANCE METRICS")
    print("=" * 40)
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    
    # Classification report
    print("\n" + "=" * 60)
    print("📋 CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('evaluation_confusion_matrix.png', dpi=150)
    plt.show()
    print("\n💾 Confusion matrix saved to 'evaluation_confusion_matrix.png'")
    
    # Per-class accuracy
    print("\n" + "=" * 40)
    print("📈 PER-CLASS ACCURACY")
    print("=" * 40)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    for name, acc in zip(CLASS_NAMES, per_class_acc):
        bar = '█' * int(acc * 20)
        spaces = ' ' * (20 - len(bar))
        print(f"{name:12} [{bar}{spaces}] {acc*100:.2f}%")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }


if __name__ == '__main__':
    results = evaluate_model()
    print("\n" + "=" * 60)
    print("✅ Evaluation completed!")
    print("=" * 60)
