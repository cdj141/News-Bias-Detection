# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 02:42:24 2025

@author: 陈东杰
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from gensim.models import KeyedVectors
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
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

# Map labels from 1-4 to 0-3
label_mapping = {label: idx for idx, label in enumerate(sorted(y.unique()))}
y = y.map(label_mapping)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# Tokenize text data
def tokenize(text):
    return text.split()

X_train_tokens = X_train.apply(tokenize)
X_test_tokens = X_test.apply(tokenize)

# Load pre-trained Word2Vec embeddings (Google News vectors)
pretrained_model_path = 'GoogleNews-vectors-negative300.bin'  # Ensure the file is in the correct location
word2vec_model = KeyedVectors.load_word2vec_format(pretrained_model_path, binary=True)

# Create document embeddings using pre-trained Word2Vec
def get_document_vector(tokens, model):
    vectors = [model[word] for word in tokens if word in model]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

X_train_embeddings = np.array([get_document_vector(tokens, word2vec_model) for tokens in X_train_tokens])
X_test_embeddings = np.array([get_document_vector(tokens, word2vec_model) for tokens in X_test_tokens])

# Define LSTM model
class LSTM_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(LSTM_Model, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Take the last hidden state
        x = self.fc(x)
        return x

# Define GRU model
class GRU_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(GRU_Model, self).__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x, _ = self.gru(x)
        x = x[:, -1, :]  # Take the last hidden state
        x = self.fc(x)
        return x

# Train and evaluate LSTM
print("\nTraining LSTM Model...")
lstm_model = LSTM_Model(input_dim=300, hidden_dim=128, num_classes=len(label_mapping))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(lstm_model.parameters(), lr=1e-4)

X_train_lstm = torch.tensor(X_train_embeddings, dtype=torch.float32).unsqueeze(1)
X_test_lstm = torch.tensor(X_test_embeddings, dtype=torch.float32).unsqueeze(1)
y_train_lstm = torch.tensor(y_train.values, dtype=torch.long)
y_test_lstm = torch.tensor(y_test.values, dtype=torch.long)

for epoch in range(20):
    lstm_model.train()
    optimizer.zero_grad()
    outputs = lstm_model(X_train_lstm)
    loss = criterion(outputs, y_train_lstm)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch + 1}/20, Loss: {loss.item():.4f}")

lstm_model.eval()
with torch.no_grad():
    outputs = lstm_model(X_test_lstm)
    _, predictions = torch.max(outputs, 1)
    accuracy = accuracy_score(y_test_lstm, predictions)
    print(f"\nLSTM Accuracy: {accuracy:.4f}")
    print("LSTM Classification Report:")
    print(classification_report(y_test_lstm, predictions))

    # Plot LSTM confusion matrix
    print("\nLSTM Confusion Matrix:")
    lstm_cm = confusion_matrix(y_test_lstm, predictions)
    print(lstm_cm)
    ConfusionMatrixDisplay(lstm_cm, display_labels=label_mapping.keys()).plot(cmap='Blues')
    plt.title("LSTM Confusion Matrix")
    plt.show()

# Train and evaluate GRU
print("\nTraining GRU Model...")
gru_model = GRU_Model(input_dim=300, hidden_dim=128, num_classes=len(label_mapping))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(gru_model.parameters(), lr=1e-4)

for epoch in range(20):
    gru_model.train()
    optimizer.zero_grad()
    outputs = gru_model(X_train_lstm)
    loss = criterion(outputs, y_train_lstm)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch + 1}/20, Loss: {loss.item():.4f}")

gru_model.eval()
with torch.no_grad():
    outputs = gru_model(X_test_lstm)
    _, predictions = torch.max(outputs, 1)
    accuracy = accuracy_score(y_test_lstm, predictions)
    print(f"\nGRU Accuracy: {accuracy:.4f}")
    print("GRU Classification Report:")
    print(classification_report(y_test_lstm, predictions))

    # Plot GRU confusion matrix
    print("\nGRU Confusion Matrix:")
    gru_cm = confusion_matrix(y_test_lstm, predictions)
    print(gru_cm)
    ConfusionMatrixDisplay(gru_cm, display_labels=label_mapping.keys()).plot(cmap='Blues')
    plt.title("GRU Confusion Matrix")
    plt.show()
