# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 01:46:30 2025

@author: 陈东杰
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
import re

# Load the dataset
processed_file_path = 'voted.csv'
data = pd.read_csv(processed_file_path)

# Function to clean sentences
def clean_sentence(sentence):
    # Remove content in square brackets and standalone digits
    sentence = re.sub(r'\[\d+\]', '', sentence)  # Remove patterns like [123]
    sentence = re.sub(r'\d+', '', sentence)  # Remove standalone digits
    return sentence.strip()

# Apply cleaning to the 'sentence' column
data['cleaned_sentence'] = data['sentence'].apply(clean_sentence)

# Extract cleaned features and labels
X = data['cleaned_sentence']
y = data['voted_label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# Convert cleaned text data to TF-IDF features
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=1),
    'Support Vector Machine': SVC(kernel='linear', random_state=1),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=1)
}

# Train and evaluate each model
for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    model.fit(X_train_tfidf, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_tfidf)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Accuracy for {model_name}: {accuracy:.4f}")
    print("Classification Report:")
    print(report)

    # Plot confusion matrix with clearer visualization
    fig, ax = plt.subplots(figsize=(10, 8))  # Adjust the figure size for better clarity
    ConfusionMatrixDisplay.from_estimator(model, X_test_tfidf, y_test, display_labels=model.classes_, cmap='Blues', ax=ax)
    plt.title(f'Confusion Matrix: {model_name}', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

