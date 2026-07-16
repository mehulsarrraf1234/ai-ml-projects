"""
Data Loading and Preprocessing for Image Classification
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image_dataset_from_directory
import os


# CIFAR-10 class names
CIFAR10_LABELS = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def load_cifar10():
    """
    Load and preprocess CIFAR-10 dataset.
    
    Returns:
        Tuple of (x_train, y_train), (x_test, y_test)
    """
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Convert labels to categorical
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    return (x_train, y_train), (x_test, y_test)


def load_cifar100():
    """
    Load and preprocess CIFAR-100 dataset.
    
    Returns:
        Tuple of (x_train, y_train), (x_test, y_test)
    """
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    
    # Normalize pixel values
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Convert labels to categorical
    y_train = to_categorical(y_train, 100)
    y_test = to_categorical(y_test, 100)
    
    return (x_train, y_train), (x_test, y_test)


def get_data_augmentation():
    """
    Create data augmentation pipeline for training.
    
    Returns:
        Data augmentation Sequential model
    """
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomFlip("vertical"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
        layers.RandomTranslation(0.1, 0.1),
    ])
    return data_augmentation


def load_custom_dataset(data_dir, image_size=(32, 32), batch_size=32):
    """
    Load custom image dataset from directory.
    
    Expected directory structure:
        data_dir/
            class1/
                img1.jpg
                img2.jpg
            class2/
                img1.jpg
                img2.jpg
    
    Args:
        data_dir: Path to data directory
        image_size: Target image size
        batch_size: Batch size for datasets
    
    Returns:
        train_ds, val_ds: Training and validation datasets
    """
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} does not exist. Using CIFAR-10 instead.")
        return None, None
    
    train_ds = image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=image_size,
        batch_size=batch_size
    )
    
    val_ds = image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=image_size,
        batch_size=batch_size
    )
    
    return train_ds, val_ds


def preprocess_images(images, labels):
    """
    Preprocess images for model input.
    
    Args:
        images: Batch of images
        labels: Batch of labels
    
    Returns:
        Preprocessed images and labels
    """
    # Normalize to [0, 1]
    images = tf.cast(images, tf.float32) / 255.0
    return images, labels


def create_tf_datasets(x_train, y_train, x_test, y_test, batch_size=64):
    """
    Create TensorFlow datasets from numpy arrays.
    
    Args:
        x_train, y_train: Training data
        x_test, y_test: Test data
        batch_size: Batch size
    
    Returns:
        train_ds, test_ds: TensorFlow datasets
    """
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train_ds, test_ds


def get_class_names(dataset_name='cifar10'):
    """
    Get class names for different datasets.
    
    Args:
        dataset_name: Name of the dataset
    
    Returns:
        List of class names
    """
    if dataset_name == 'cifar10':
        return CIFAR10_LABELS
    elif dataset_name == 'cifar100':
        return [f'class_{i}' for i in range(100)]
    else:
        return None
