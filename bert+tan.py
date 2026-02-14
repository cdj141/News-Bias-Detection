# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 03:35:59 2025

@author: 陈东杰
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from transformers import BertTokenizer, BertModel
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

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Create embeddings for Transformer
def get_bert_embeddings(texts, tokenizer, model):
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).numpy()
        embeddings.append(cls_embedding)
    return np.array(embeddings)

X_train_embeddings = get_bert_embeddings(X_train.tolist(), tokenizer, bert_model)
X_test_embeddings = get_bert_embeddings(X_test.tolist(), tokenizer, bert_model)

# Define Transformer model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, num_classes):
        super(TransformerModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = x.transpose(0, 1)  # Change shape to (seq_length, batch_size, input_dim)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)  # Aggregate over sequence dimension
        x = self.fc(x)
        return x

# Adjust input shape for Transformer
X_train_tensor = torch.tensor(X_train_embeddings, dtype=torch.float32).unsqueeze(1)  # Add sequence length dimension
X_test_tensor = torch.tensor(X_test_embeddings, dtype=torch.float32).unsqueeze(1)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

# Train and evaluate Transformer
print("\nTraining Transformer Model...")
transformer_model = TransformerModel(input_dim=768, num_heads=4, num_layers=2, num_classes=len(label_mapping))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(transformer_model.parameters(), lr=1e-4)

for epoch in range(200):
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

