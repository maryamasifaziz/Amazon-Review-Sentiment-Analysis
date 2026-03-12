import pandas as pd
import numpy as np
from datasets import load_dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import os

# Loading Dataset
print("Loading dataset...")
dataset = load_dataset("amazon_polarity", split="train[:20000]")
df = pd.DataFrame(dataset)

# Keeping only relevant columns
df = df[['content', 'label']]
df.columns = ['text', 'label']  # 0 = negative, 1 = positive

print(f"Dataset shape: {df.shape}")
print(df['label'].value_counts())

# Train-Test Split Data 
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].tolist(),
    df['label'].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

print(f"Train size: {len(train_texts)}, Val size: {len(val_texts)}")

# Tokenizer 
print("Loading BERT tokenizer...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize(texts, labels):
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors='pt'
    )
    return encodings, torch.tensor(labels)

train_encodings, train_label_tensor = tokenize(train_texts, train_labels)
val_encodings, val_label_tensor = tokenize(val_texts, val_labels)

# PyTorch Dataset
class ReviewDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

train_dataset = ReviewDataset(train_encodings, train_label_tensor)
val_dataset = ReviewDataset(val_encodings, val_label_tensor)

# Loading BERT Model 
print("Loading BERT model...")
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2
)

# Training Arguments 
training_args = TrainingArguments(
    output_dir='./models',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

print("Training started...")
trainer.train()

# Saving model
model.save_pretrained('./models/bert-sentiment')
tokenizer.save_pretrained('./models/bert-sentiment')
print("Model saved!")