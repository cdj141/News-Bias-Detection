# News Bias Detection

Sentence-level bias classification in news articles using traditional
machine learning and deep learning models.

## Overview

This project evaluates the impact of different text representation
techniques and classification models on multi-class news bias detection.

Compared approaches:

-   TF-IDF + Logistic Regression / SVM / Random Forest
-   Word2Vec + LSTM / GRU / Transformer
-   BERT + LSTM / GRU / Transformer

The objective is to analyze how contextual embeddings influence
classification performance under class imbalance.

## Dataset

-   837 unique sentences
-   4 bias classes: 1 -- Neutral 2 -- Slightly Biased 3 -- Biased 4 --
    Very Biased
-   Strong class imbalance (Class 3 dominates)

Data was processed using majority voting and stratified 70/30 train-test
split.

## Data Preprocessing

-   Removed bracketed content (e.g., \[123\])
-   Removed standalone numeric tokens
-   Deduplicated sentences
-   Stratified sampling

Script: vote.py

## Text Representations

TF-IDF\
- Max features: 5000\
- Used with traditional ML models

Word2Vec\
- Pretrained Google News embeddings (300d)\
- Sentence vector = mean of word vectors

BERT (bert-base-uncased)\
- CLS token embeddings (768d)\
- Contextualized representations

## Models

Traditional Machine Learning: - Logistic Regression - Support Vector
Machine (Linear kernel) - Random Forest

Deep Learning: - Bidirectional LSTM (128 hidden units) - Bidirectional
GRU (128 hidden units) - Transformer (4 heads, 2 encoder layers)

Training: - Optimizer: Adam - Learning rate: 1e-4 - CrossEntropyLoss

## Results

Best Traditional Model: SVM + TF-IDF\
Accuracy: 56.75%

Best Overall Model: BERT + Transformer\
Accuracy: 58.73%\
Weighted F1: 0.56

Contextual embeddings significantly outperform static embeddings.

## Limitations

-   Small dataset size
-   Severe class imbalance
-   High computational cost for BERT-based models
-   Poor minority class detection

## Future Improvements

-   Class weighting or focal loss
-   Oversampling minority classes
-   Domain-specific BERT fine-tuning
-   Data augmentation
-   Ensemble strategies

## Repository Structure

vote.py\
tf-idf ml.py\
woed2vec lstm+gru.py\
woed2vec+tran.py\
bert+lstm+gru.py\
bert+tan.py

## Run

pip install -r requirements.txt\
python tf-idf ml.py

## Technologies

Python\
PyTorch\
scikit-learn\
Transformers (HuggingFace)\
Gensim\
NumPy / Pandas / Matplotlib

## Author

Dongjie Chen\
MSc Computer Science (Data Science), Leiden University
