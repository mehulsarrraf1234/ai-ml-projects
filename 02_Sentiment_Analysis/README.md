# 💬 Sentiment Analysis System

An NLP pipeline for analyzing sentiment in text data using traditional ML and transformer-based approaches.

## 📌 Project Overview

This project implements a complete sentiment analysis system with support for:
- Traditional ML approaches (TF-IDF + Classifiers)
- Deep Learning models (LSTM, CNN)
- Pre-trained transformers (BERT, DistilBERT)
- Real-time prediction via Flask API

## 🔧 Features

- Text preprocessing pipeline (tokenization, stemming, lemmatization)
- TF-IDF vectorization
- Word embeddings (Word2Vec, GloVe)
- Multiple model architectures (Logistic Regression, SVM, LSTM, BERT)
- Model evaluation and comparison
- Flask API for real-time predictions
- Interactive Streamlit dashboard

## 📊 Datasets

- IMDB Movie Reviews Dataset (25,000 reviews)
- Twitter Sentiment Analysis Dataset
- Amazon Product Reviews

## 🛠️ Installation

```bash
pip install tensorflow keras nltk scikit-learn pandas numpy flask streamlit
```

Or use the provided requirements.txt:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Train models:
```bash
python train.py
```

### Run Flask API:
```bash
python api.py
```

### Run Streamlit dashboard:
```bash
streamlit run dashboard.py
```

### Make predictions:
```bash
python predict.py --text "This movie was amazing!"
```

## 📁 Project Structure

```
02_Sentiment_Analysis/
├── train.py            # Model training script
├── api.py              # Flask API for predictions
├── dashboard.py        # Streamlit dashboard
├── predict.py          # Command-line prediction
├── models/
│   ├── traditional.py  # Traditional ML models
│   ├── deep_learning.py # LSTM, CNN models
│   └── transformers.py  # BERT models
├── preprocessing.py    # Text preprocessing
├── evaluation.py       # Model evaluation
├── requirements.txt    # Dependencies
└── README.md          # This file
```

## 🎓 Learning Outcomes

- Text preprocessing and NLP fundamentals
- Feature extraction (TF-IDF, embeddings)
- Sentiment classification techniques
- Model deployment via APIs
- Transformer-based models (BERT)

## 👤 Author

**Mehul Sarraf**  
- GitHub: [@mehulsarrraf1234](https://github.com/mehulsarrraf1234)
- LinkedIn: [Mehul Sarraf](https://www.linkedin.com/in/mehul-sarraf-3a3b11313)

## 📝 License

This project is open source and available under the MIT License.
