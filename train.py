import pandas as pd
import re
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
import os
import pickle

# Create the models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Load dataset
df = pd.read_csv('mnt/data/updated_code_snippets.csv')
print(df.columns)

# Preprocess the code snippets
def preprocess_text(text):
    text = re.sub(r'[^\w\s\{\}\(\)\[\];:,.<>]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower().strip()
    return text

df['cleaned_snippet'] = df['python_solutions'].apply(preprocess_text)

# Additional features
df['num_lines'] = df['python_solutions'].apply(lambda x: len(x.split('\n')))
df['avg_line_length'] = df['python_solutions'].apply(lambda x: np.mean([len(line) for line in x.split('\n')]))
df['num_comments'] = df['python_solutions'].apply(lambda x: len(re.findall(r'//|#', x)))
df['indentation_consistency'] = df['python_solutions'].apply(lambda x: len(set(re.findall(r'^\s+', x, re.MULTILINE))))
df['cyclomatic_complexity'] = df['python_solutions'].apply(lambda x: len(re.findall(r'\b(if|for|while|case|catch)\b', x)))
df['unbalanced_brackets'] = df['python_solutions'].apply(lambda x: abs(x.count('(') - x.count(')')) + abs(x.count('{') - x.count('}')))
df['contains_digits'] = df['python_solutions'].apply(lambda x: int(bool(re.search(r'\d', x))))
df['avg_word_length'] = df['python_solutions'].apply(lambda x: np.mean([len(word) for word in x.split()]))
df['num_functions'] = df['python_solutions'].apply(lambda x: len(re.findall(r'\bdef\b|\bfunction\b|\bvoid\b', x)))

# Print readability value counts
print("Original readability counts:")
print(df['readability'].value_counts())

# Convert readability labels to binary using the median as the threshold
threshold = df['readability'].median()
df['readability'] = df['readability'].apply(lambda x: 1 if x >= threshold else 0)
print("Converted readability counts:")
print(df['readability'].value_counts())

# Ensure balanced dataset
min_count = min(df['readability'].value_counts())
df_readable = df[df['readability'] == 1].sample(min_count, random_state=42)
df_not_readable = df[df['readability'] == 0].sample(min_count, random_state=42)
df_balanced = pd.concat([df_readable, df_not_readable])

# Vectorize the code snippets
vectorizer = TfidfVectorizer(max_features=5000)
X_text = vectorizer.fit_transform(df_balanced['cleaned_snippet']).toarray()
X_additional = df_balanced[['num_lines', 'avg_line_length', 'num_comments', 'indentation_consistency', 'cyclomatic_complexity', 'unbalanced_brackets', 'contains_digits', 'avg_word_length', 'num_functions']].values
X = np.hstack((X_text, X_additional))
y = df_balanced['readability'].values

# Save the vectorizer
with open('models/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# Get the actual number of features
input_dim = X.shape[1]
print(f"Actual number of features: {input_dim}")

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model with more layers or different architecture
model = Sequential([
    Dense(128, input_shape=(input_dim,), activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

# Save the model weights
model.save_weights('models/model_weights.weights.h5')
