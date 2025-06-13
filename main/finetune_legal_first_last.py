import numpy as np
import pandas as pd
import time
import datetime
import random
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
import sys
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse

import warnings
warnings.filterwarnings("ignore", message=".*overflowing tokens are not returned.*")

import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.cuda.manual_seed_all(42)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Load dataset
def load_data(filename):
    df = pd.read_csv(filename)
    return df['case_text'], df['case_label']

# Tokenize cases for default truncation
# def tokenize_cases(cases, labels, tokenizer, max_len):
#     input_ids = []
#     attention_masks = []
    
#     for case in cases:
#         encoded_dict = tokenizer.encode_plus(
#             case,
#             add_special_tokens=True,
#             max_length=max_len,
#             padding='max_length',
#             truncation='longest_first',
#             return_attention_mask=True,
#             return_tensors='pt',
#             return_overflowing_tokens=False
#         )
#         input_ids.append(encoded_dict['input_ids'])
#         attention_masks.append(encoded_dict['attention_mask'])
    
#     input_ids = torch.cat(input_ids, dim=0)
#     attention_masks = torch.cat(attention_masks, dim=0)
#     labels = torch.tensor(labels)
    
#     return input_ids, attention_masks, labels


def tokenize_cases(cases, labels, tokenizer, max_len):
    input_ids = []
    attention_masks = []

    for case in cases:
        encoded = tokenizer.encode_plus(
            case,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=False
        )

        tokens = encoded['input_ids'][0]  # shape: [seq_len]

        if len(tokens) > max_len:
            first_part = tokens[:256]
            last_part = tokens[-256:]
            tokens = torch.cat([first_part, last_part])

            # Replace with special tokens if applicable
            if tokenizer.cls_token_id is not None:
                tokens[0] = tokenizer.cls_token_id
            if tokenizer.sep_token_id is not None and len(tokens) >= 2:
                tokens[-1] = tokenizer.sep_token_id

            attention_mask = torch.ones_like(tokens)

        else:
            padding_len = max_len - len(tokens)
            pad_tensor = torch.full((padding_len,), tokenizer.pad_token_id, dtype=torch.long)
            tokens = torch.cat([tokens, pad_tensor])

            attention_mask = torch.cat([
                torch.ones(len(tokens) - padding_len, dtype=torch.long),
                torch.zeros(padding_len, dtype=torch.long)
            ])

        input_ids.append(tokens.unsqueeze(0))
        attention_masks.append(attention_mask.unsqueeze(0))

    # Final concat and type fixes
    input_ids = torch.cat(input_ids, dim=0).type(torch.long)
    attention_masks = torch.cat(attention_masks, dim=0).type(torch.long)
    labels = torch.tensor(labels, dtype=torch.long)

    return input_ids, attention_masks, labels


# Create DataLoader
def create_dataloaders(train_dataset, dev_dataset, batch_size):
    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=batch_size
    )
    
    val_dataloader = DataLoader(
        dev_dataset,
        sampler=SequentialSampler(dev_dataset),
        batch_size=batch_size
    )
    
    return train_dataloader, val_dataloader

# Train the model
def train_model(model, optimizer, scheduler, train_dataloader, val_dataloader, outname, epochs):
    total_t0 = time.time()
    training_stats = []
    best_eval_accuracy = 0
    
    for epoch_i in range(0, epochs):
        print(f'\n======== Epoch {epoch_i + 1} / {epochs} ========')
        print('Training...')
        t0 = time.time()
        total_train_loss = 0
        model.train()
        
        for batch in tqdm(train_dataloader):
            b_input_ids, b_input_mask, b_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            optimizer.zero_grad()
            output = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            loss = output.loss
            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            scheduler.step()
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        training_time = format_time(time.time() - t0)
        print(f"  Average training loss: {avg_train_loss:.2f}")
        print(f"  Training epoch took: {training_time}")
        
        print("\nRunning Validation...")
        t0 = time.time()
        model.eval()
        total_eval_accuracy = 0
        total_eval_loss = 0
        
        
        for batch in tqdm(val_dataloader):
            b_input_ids, b_input_mask, b_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            with torch.no_grad():
                output = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            loss = output.loss
            total_eval_loss += loss.item()
            logits = output.logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            total_eval_accuracy += flat_accuracy(logits, label_ids)
        
        avg_val_accuracy = total_eval_accuracy / len(val_dataloader)
        avg_val_loss = total_eval_loss / len(val_dataloader)
        validation_time = format_time(time.time() - t0)
        if avg_val_accuracy > best_eval_accuracy:
            torch.save(model, outname)
            best_eval_accuracy = avg_val_accuracy
        
        print(f"  Accuracy: {avg_val_accuracy:.2f}")
        print(f"  Validation Loss: {avg_val_loss:.2f}")
        print(f"  Validation took: {validation_time}")
        
        training_stats.append({
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        })
    
    print("\nTraining complete!")
    print(f"Total training took {format_time(time.time() - total_t0)}")
    return training_stats

def evaluate(model,dataloader):
    predictions = []
    total_eval_accuracy=0
    for batch in tqdm(dataloader):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels=batch[2].to(device)
        with torch.no_grad():        
            output= model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            logits = output.logits
            logits = logits.detach().cpu().numpy()
            label_ids=b_labels.to('cpu').numpy()
        total_eval_accuracy += flat_accuracy(logits, label_ids)
    avg_test_accuracy = total_eval_accuracy / len(dataloader)
    return avg_test_accuracy

# Utility functions
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    elapsed_rounded = int(round(elapsed))
    return str(datetime.timedelta(seconds=elapsed_rounded))

# Main function
def main():

    batch_size = 32
    epochs = 5
    max_len = 512
    lr = 2e-5
    weight_decay=0.01
    
    train_filename='dataset/legal_text_classifcation_train.csv'
    val_filename='dataset/legal_text_classifcation_val.csv'
    test_filename='dataset/legal_text_classifcation_test.csv'

    model_name = "nlpaueb/legal-bert-base-uncased"
    outname = "saved_models/legalbert_trained_first_last.pt"
    
    train_cases, train_labels = load_data(train_filename)
    val_cases, val_labels = load_data(val_filename)
    test_cases, test_labels = load_data(test_filename)
    # Load tokenizer and tokenize data
    tokenizer = BertTokenizer.from_pretrained(model_name)
    train_input_ids, train_attention_mask, train_labels = tokenize_cases(train_cases, train_labels, tokenizer,max_len)
    dev_input_ids, dev_attention_mask, dev_labels = tokenize_cases(val_cases, val_labels, tokenizer,max_len)
    test_input_ids, test_attention_mask, test_labels = tokenize_cases(test_cases, test_labels, tokenizer,max_len)
    # Create datasets
    train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_labels)
    dev_dataset = TensorDataset(dev_input_ids, dev_attention_mask, dev_labels)
    test_dataset = TensorDataset(test_input_ids, test_attention_mask, test_labels)
    # Create dataloaders
    train_dataloader, val_dataloader = create_dataloaders(train_dataset, dev_dataset,32)
    test_dataloader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=batch_size
    )

    print("Loading Model")
    model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=10,
            ignore_mismatched_sizes=True 
    )

    model = model.to(device)
    
    # Set up optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8, weight_decay=weight_decay)
    total_steps = len(train_dataloader) * epochs 
    
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
    
    # Train the model
    training_stats = train_model(model, optimizer, scheduler, train_dataloader, val_dataloader,outname,epochs)
    # Test the trained model
    model = torch.load(outname, weights_only=False)
    test_stats=evaluate(model,test_dataloader)
    print("Accuracy:", test_stats)
    return training_stats

if __name__ == "__main__":
    main()
