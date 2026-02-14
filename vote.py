# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 01:38:08 2025

@author: 陈东杰
"""

import pandas as pd

# Load the original dataset
file_path = 'Sora_LREC2020_biasedsentences.csv'
data = pd.read_csv(file_path)

# Combine sentences into full articles
sentence_columns = [col for col in data.columns if col.startswith('s') and col[1:].isdigit()]
data['combined_text'] = data[sentence_columns].apply(
    lambda row: ' '.join(row.dropna()), axis=1
)

# Group by 'combined_text' and apply majority voting for the labels
annotator_data = data[['combined_text', 'article_bias']].copy()
annotator_data['annotator_id'] = annotator_data.groupby('combined_text').cumcount() + 1
voted_data = annotator_data.groupby('combined_text')['article_bias'].agg(lambda x: x.value_counts().idxmax()).reset_index()
voted_data.rename(columns={'article_bias': 'voted_label'}, inplace=True)

# Expand the data to keep individual sentences along with their voted labels
expanded_data = voted_data.merge(data[sentence_columns + ['combined_text']], on='combined_text', how='left')
expanded_data_melted = expanded_data.melt(
    id_vars=['combined_text', 'voted_label'], 
    value_vars=sentence_columns, 
    var_name='sentence_id', 
    value_name='sentence'
).dropna(subset=['sentence'])

# Remove duplicates to keep only one occurrence of each sentence
unique_sentences = expanded_data_melted.drop_duplicates(subset=['sentence'])

# Save the deduplicated data to a CSV file
unique_sentences_file_path = 'voted.csv'
unique_sentences.to_csv(unique_sentences_file_path, index=False)

print(f"Processed data saved to: {unique_sentences_file_path}")
