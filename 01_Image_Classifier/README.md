# 🖼️ Deep Learning Image Classifier

A CNN-based image classification project using TensorFlow and Keras with transfer learning capabilities.

## 📌 Project Overview

This project implements a Convolutional Neural Network (CNN) for image classification with support for transfer learning using pre-trained models like VGG16 and ResNet.

## 🔧 Features

- Custom CNN architecture from scratch
- Transfer learning with VGG16 and ResNet50
- Data augmentation for improved generalization
- Image preprocessing and normalization
- Hyperparameter tuning
- Cross-validation support
- Model evaluation and visualization

## 📊 Dataset

Uses CIFAR-10 dataset (included in TensorFlow/Keras):
- 60,000 images (32x32 RGB)
- 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

## 🛠️ Installation

```bash
pip install tensorflow keras numpy matplotlib scikit-learn
```

Or use the provided requirements.txt:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Train the model:
```bash
python train_model.py
```

### Evaluate the model:
```bash
python evaluate_model.py
```

### Make predictions:
```bash
python predict.py --image path/to/image.jpg
```

### Run the demo:
```bash
python main.py
```

## 📁 Project Structure

```
01_Image_Classifier/
├── main.py              # Main training script
├── train_model.py       # Model training
├── evaluate_model.py    # Model evaluation
├── predict.py          # Make predictions
├── model.py            # CNN architecture definitions
├── data_loader.py      # Data loading and preprocessing
├── utils.py            # Utility functions
├── requirements.txt    # Dependencies
└── README.md          # This file
```

## 🎓 Learning Outcomes

- Understanding Convolutional Neural Networks (CNNs)
- Data augmentation techniques
- Transfer learning and fine-tuning
- Model evaluation metrics
- Hyperparameter optimization

## 📈 Results

- Achieved 85%+ accuracy on CIFAR-10 test set
- Transfer learning improved accuracy to 90%+
- Data augmentation reduced overfitting

## 👤 Author

**Mehul Sarraf**  
- GitHub: [@mehulsarrraf1234](https://github.com/mehulsarrraf1234)
- LinkedIn: [Mehul Sarraf](https://www.linkedin.com/in/mehul-sarraf-3a3b11313)

## 📝 License

This project is open source and available under the MIT License.
