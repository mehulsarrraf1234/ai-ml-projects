"""
Main Training Script for Deep Learning Image Classifier
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

from model import create_custom_cnn, create_vgg16_transfer, create_resnet_transfer, compile_model
from data_loader import load_cifar10, get_data_augmentation, create_tf_datasets


def plot_training_history(history):
    """
    Plot training and validation accuracy/loss curves.
    
    Args:
        history: Keras History object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
    ax1.set_title('Model Accuracy', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Loss plot
    ax2.plot(history.history['loss'], label='Training Loss', marker='o')
    ax2.plot(history.history['val_loss'], label='Validation Loss', marker='o')
    ax2.set_title('Model Loss', fontsize=14)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    plt.show()
    print("Training history saved to 'training_history.png'")


def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=14)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.show()
    print("Confusion matrix saved to 'confusion_matrix.png'")


def plot_sample_predictions(x_test, y_test, model, class_names, num_samples=16):
    """
    Plot sample predictions from the test set.
    
    Args:
        x_test: Test images
        y_test: True labels
        model: Trained model
        class_names: List of class names
        num_samples: Number of samples to plot
    """
    # Get predictions
    predictions = model.predict(x_test[:num_samples])
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(y_test[:num_samples], axis=1)
    
    # Create figure
    fig, axes = plt.subplots(4, 4, figsize=(14, 14))
    axes = axes.ravel()
    
    for i in range(num_samples):
        axes[i].imshow(x_test[i])
        color = 'green' if predicted_labels[i] == true_labels[i] else 'red'
        title = f'True: {class_names[true_labels[i]]}\nPred: {class_names[predicted_labels[i]]}'
        axes[i].set_title(title, color=color, fontsize=10)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_predictions.png', dpi=150)
    plt.show()
    print("Sample predictions saved to 'sample_predictions.png'")


def train_model(model_type='custom', epochs=50, batch_size=64, use_augmentation=True):
    """
    Train the image classification model.
    
    Args:
        model_type: 'custom', 'vgg16', or 'resnet'
        epochs: Number of training epochs
        batch_size: Batch size for training
        use_augmentation: Whether to use data augmentation
    
    Returns:
        Trained model and training history
    """
    print("=" * 60)
    print(f"Training {model_type.upper()} CNN Image Classifier")
    print("=" * 60)
    
    # Load data
    print("\n📂 Loading CIFAR-10 dataset...")
    (x_train, y_train), (x_test, y_test) = load_cifar10()
    print(f"Training samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")
    
    # Create model
    print(f"\n🏗️ Creating {model_type} model...")
    if model_type == 'custom':
        model = create_custom_cnn()
    elif model_type == 'vgg16':
        model = create_vgg16_transfer()
    elif model_type == 'resnet':
        model = create_resnet_transfer()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = compile_model(model)
    print("Model created and compiled successfully!")
    
    # Setup callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True
        )
    ]
    
    # Create datasets
    print("\n🔄 Preparing datasets...")
    train_ds, test_ds = create_tf_datasets(x_train, y_train, x_test, y_test, batch_size)
    
    # Train model
    print(f"\n🚀 Starting training for {epochs} epochs...")
    print("-" * 40)
    
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=test_ds,
        callbacks=callbacks if use_augmentation else callbacks[:2]
    )
    
    # Evaluate
    print("\n📊 Evaluating model on test set...")
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Plot results
    print("\n📈 Generating visualizations...")
    plot_training_history(history)
    
    # Confusion matrix
    predictions = model.predict(x_test)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    plot_confusion_matrix(y_true, y_pred, [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ])
    
    # Classification report
    print("\n📋 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=[
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]))
    
    # Sample predictions
    plot_sample_predictions(x_test, y_test, model, [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ])
    
    # Save model
    model.save('image_classifier_model.h5')
    print("\n✅ Model saved to 'image_classifier_model.h5'")
    
    return model, history


if __name__ == '__main__':
    import tensorflow as tf
    
    # Train with custom CNN (can change to 'vgg16' or 'resnet')
    model, history = train_model(
        model_type='custom',    # Options: 'custom', 'vgg16', 'resnet'
        epochs=30,              # Adjust based on your time/resources
        batch_size=64,
        use_augmentation=True
    )
    
    print("\n" + "=" * 60)
    print("🎉 Training completed successfully!")
    print("=" * 60)
