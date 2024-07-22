import re
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from flask import Flask, request, render_template, redirect, url_for
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load the vectorizer
def load_vectorizer():
    with open('models/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return vectorizer

# Load the trained model
def load_model(input_dim):
    model = Sequential([
        Dense(128, input_shape=(input_dim,), activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.load_weights('models/model_weights.weights.h5')
    return model

# Preprocess input code snippet
def preprocess(snippet):
    snippet = re.sub(r'[^\w\s\{\}\(\)\[\];:,.<>]', ' ', snippet)
    snippet = re.sub(r'\s+', ' ', snippet)
    snippet = snippet.lower().strip()
    return snippet

# Additional features
def extract_features(snippet):
    num_lines = len(snippet.split('\n'))
    avg_line_length = np.mean([len(line) for line in snippet.split('\n')])
    num_comments = len(re.findall(r'//|#', snippet))
    indentation_consistency = len(set(re.findall(r'^\s+', snippet, re.MULTILINE)))
    cyclomatic_complexity = len(re.findall(r'\b(if|for|while|case|catch)\b', snippet))
    unbalanced_brackets = abs(snippet.count('(') - snippet.count(')')) + abs(snippet.count('{') - snippet.count('}'))
    contains_digits = int(bool(re.search(r'\d', snippet)))
    avg_word_length = np.mean([len(word) for word in snippet.split()])
    num_functions = len(re.findall(r'\bdef\b|\bfunction\b|\bvoid\b', snippet))
    return [num_lines, avg_line_length, num_comments, indentation_consistency, cyclomatic_complexity, unbalanced_brackets, contains_digits, avg_word_length, num_functions]

# Initialize vectorizer
vectorizer = load_vectorizer()
input_dim = len(vectorizer.get_feature_names_out()) + 9  # Adding 9 for additional features

# Initialize model
model = load_model(input_dim)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        snippet = request.form['snippet']
        cleaned_snippet = preprocess(snippet)
        transformed_snippet = vectorizer.transform([cleaned_snippet]).toarray()
        additional_features = np.array(extract_features(snippet)).reshape(1, -1)
        final_input = np.hstack((transformed_snippet, additional_features))
        if final_input.shape[1] != input_dim:
            return render_template('result.html', prediction="Error: Model input size mismatch.", snippet=snippet)
        prediction = model.predict(final_input)
        output = 'Readable' if prediction[0][0] > 0.5 else 'Not Readable'
        return render_template('result.html', prediction=output, snippet=snippet)

@app.route('/feedback', methods=['POST'])
def feedback():
    snippet = request.form['snippet']
    prediction = request.form['prediction']
    correct = request.form['correct']
    
    # Save the feedback for future training
    feedback_data = {'snippet': snippet, 'prediction': prediction, 'correct': correct}
    feedback_df = pd.DataFrame([feedback_data])
    
    if os.path.exists('feedback.csv'):
        feedback_df.to_csv('feedback.csv', mode='a', header=False, index=False)
    else:
        feedback_df.to_csv('feedback.csv', index=False)
    
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
