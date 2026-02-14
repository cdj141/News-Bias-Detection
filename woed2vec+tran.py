# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 03:20:52 2025

@author: 陈东杰
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
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

# Define Transformer model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, num_classes, max_seq_length):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, input_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_length, input_dim))
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        seq_length = x.size(1)
        x = self.embedding(x) + self.positional_encoding[:, :seq_length, :]
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.fc(x)
        return x

# Create embeddings for Transformer
pretrained_model_path = 'GoogleNews-vectors-negative300.bin'  # Ensure the file is in the correct location
from gensim.models import KeyedVectors
word2vec_model = KeyedVectors.load_word2vec_format(pretrained_model_path, binary=True)

def get_sequence_embeddings(tokens_list, model, max_seq_length, embedding_dim):
    embeddings = []
    for tokens in tokens_list:
        seq_embeddings = [model[word] for word in tokens if word in model]
        if len(seq_embeddings) > max_seq_length:
            seq_embeddings = seq_embeddings[:max_seq_length]
        elif len(seq_embeddings) < max_seq_length:
            seq_embeddings += [np.zeros(embedding_dim)] * (max_seq_length - len(seq_embeddings))
        embeddings.append(seq_embeddings)
    return np.array(embeddings)

max_seq_length = 50  # Maximum sequence length
embedding_dim = 300

X_train_embeddings = get_sequence_embeddings(X_train_tokens, word2vec_model, max_seq_length, embedding_dim)
X_test_embeddings = get_sequence_embeddings(X_test_tokens, word2vec_model, max_seq_length, embedding_dim)

# Train and evaluate Transformer
print("\nTraining Transformer Model...")
transformer_model = TransformerModel(input_dim=embedding_dim, num_heads=4, num_layers=2, num_classes=len(label_mapping), max_seq_length=max_seq_length)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(transformer_model.parameters(), lr=1e-4)

X_train_tensor = torch.tensor(X_train_embeddings, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_embeddings, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

for epoch in range(100):
    transformer_model.train()
    optimizer.zero_grad()
    outputs = transformer_model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch + 1}/20, Loss: {loss.item():.4f}")

transformer_model.eval()
with torch.no_grad():
    outputs = transformer_model(X_test_tensor)
    _, predictions = torch.max(outputs, 1)
    accuracy = accuracy_score(y_test_tensor, predictions)
    print(f"\nTransformer Accuracy: {accuracy:.4f}")
    print("Transformer Classification Report:")
    print(classification_report(y_test_tensor, predictions))

    # Plot Transformer confusion matrix
    print("\nTransformer Confusion Matrix:")
    transformer_cm = confusion_matrix(y_test_tensor, predictions)
    print(transformer_cm)
    ConfusionMatrixDisplay(transformer_cm, display_labels=label_mapping.keys()).plot(cmap='Blues')
    plt.title("Transformer Confusion Matrix")
    plt.show()
