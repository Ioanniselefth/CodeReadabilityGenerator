import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model (assuming you have already trained and saved it)
def load_model():
    model = Sequential([
        Dense(128, input_dim=5000, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.load_weights('path_to_your_model_weights.h5')
    return model

# Preprocess input code snippet
def preprocess(snippet, vectorizer):
    # Add the same preprocessing steps you used during training
    snippet = re.sub(r'\W', ' ', snippet)
    snippet = re.sub(r'\d', ' ', snippet)
    snippet = snippet.lower()
    snippet = re.sub(r'\s+', ' ', snippet)
    return vectorizer.transform([snippet]).toarray()

# Initialize model and vectorizer
model = load_model()
vectorizer = TfidfVectorizer(max_features=5000)
