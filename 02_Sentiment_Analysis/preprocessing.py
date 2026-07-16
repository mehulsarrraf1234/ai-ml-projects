"""
Text Preprocessing Pipeline for Sentiment Analysis
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)


class TextPreprocessor:
    """
    Complete text preprocessing pipeline for NLP tasks.
    """
    
    def __init__(self, use_stemming=False, use_lemmatization=True):
        """
        Initialize preprocessor.
        
        Args:
            use_stemming: Whether to use stemming
            use_lemmatization: Whether to use lemmatization
        """
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization
        
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        
        # Custom stop words for sentiment analysis
        self.custom_stop_words = {'not', 'no', 'very', 'too', 'more', 'most', 'but', 'also'}
        self.stop_words.update(self.custom_stop_words)
    
    def lowercase(self, text):
        """Convert text to lowercase."""
        return text.lower()
    
    def remove_urls(self, text):
        """Remove URLs from text."""
        return re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    def remove_html_tags(self, text):
        """Remove HTML tags from text."""
        return re.sub(r'<.*?>', '', text)
    
    def remove_emojis(self, text):
        """Remove emojis from text."""
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE
        )
        return emoji_pattern.sub(r'', text)
    
    def remove_special_chars(self, text):
        """Remove special characters and numbers."""
        return re.sub(r'[^a-zA-Z\s]', '', text)
    
    def remove_extra_whitespace(self, text):
        """Remove extra whitespace."""
        return ' '.join(text.split())
    
    def expand_contractions(self, text):
        """Expand common contractions."""
        contractions = {
            "n't": "not",
            "'re": "are",
            "'s": "is",
            "'d": "would",
            "'ll": "will",
            "'ve": "have",
            "'m": "am"
        }
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        return text
    
    def remove_repeated_chars(self, text):
        """Remove repeated characters (e.g., 'soooo' -> 'soo')."""
        return re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    def tokenize(self, text):
        """Tokenize text into words."""
        return word_tokenize(text)
    
    def remove_stopwords(self, tokens):
        """Remove stopwords from tokens."""
        return [token for token in tokens if token not in self.stop_words]
    
    def apply_stemming(self, tokens):
        """Apply stemming to tokens."""
        return [self.stemmer.stem(token) for token in tokens]
    
    def apply_lemmatization(self, tokens):
        """Apply lemmatization to tokens."""
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess(self, text):
        """
        Complete preprocessing pipeline.
        
        Args:
            text: Raw text string
        
        Returns:
            Preprocessed text string
        """
        # Basic cleaning
        text = self.lowercase(text)
        text = self.expand_contractions(text)
        text = self.remove_urls(text)
        text = self.remove_html_tags(text)
        text = self.remove_emojis(text)
        text = self.remove_special_chars(text)
        text = self.remove_repeated_chars(text)
        text = self.remove_extra_whitespace(text)
        
        # Tokenization
        tokens = self.tokenize(text)
        
        # Stopword removal
        tokens = self.remove_stopwords(tokens)
        
        # Stemming or Lemmatization
        if self.use_stemming:
            tokens = self.apply_stemming(tokens)
        elif self.use_lemmatization:
            tokens = self.apply_lemmatization(tokens)
        
        return ' '.join(tokens)
    
    def preprocess_batch(self, texts):
        """
        Preprocess a batch of texts.
        
        Args:
            texts: List of text strings
        
        Returns:
            List of preprocessed text strings
        """
        return [self.preprocess(text) for text in texts]


def load_imdb_dataset():
    """
    Load IMDB movie reviews dataset.
    Uses keras built-in dataset.
    
    Returns:
        Tuple of (train_texts, train_labels), (test_texts, test_labels)
    """
    from tensorflow.keras.datasets import imdb
    
    # Load with most common words (top 10000)
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
    
    # Convert back to text (word index to words)
    word_index = imdb.get_word_index()
    
    def decode_review(encoded_review):
        reverse_word_index = {v: k for k, v in word_index.items()}
        return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])
    
    x_train_texts = [decode_review(review) for review in x_train]
    x_test_texts = [decode_review(review) for review in x_test]
    
    return (x_train_texts, y_train), (x_test_texts, y_test)


def load_twitter_dataset(file_path=None):
    """
    Load Twitter sentiment dataset.
    
    Args:
        file_path: Path to CSV file (optional)
    
    Returns:
        DataFrame with tweets and labels
    """
    # If no file provided, generate sample data
    import pandas as pd
    
    sample_data = {
        'text': [
            "I love this product! It's amazing!",
            "This is the worst experience ever",
            "Pretty good, would recommend",
            "Not happy with the service",
            "Absolutely fantastic!",
            "Terrible quality, very disappointed",
            "It's okay, nothing special",
            "Best purchase I've ever made!",
            "Waste of money",
            "Very satisfied with my order"
        ],
        'sentiment': [1, 0, 1, 0, 1, 0, 1, 1, 0, 1]  # 1=positive, 0=negative
    }
    
    return pd.DataFrame(sample_data)


# Demo
if __name__ == '__main__':
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Sample texts
    sample_texts = [
        "I absolutely LOVE this product! It's amazing 😍 http://example.com",
        "This is the WORST service I've ever experienced!!!",
        "Pretty good, would recommend to friends.",
        "Not happy with the quality at all...",
        "The movie was fantastic! Best I've seen this year."
    ]
    
    print("=" * 60)
    print("TEXT PREPROCESSING DEMO")
    print("=" * 60)
    
    for text in sample_texts:
        print(f"\n📝 Original: {text}")
        print(f"✨ Processed: {preprocessor.preprocess(text)}")
