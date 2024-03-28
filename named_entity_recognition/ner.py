import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd


# if a word is not in your vocabulary use len(vocabulary) as the encoding
class NERDataset(Dataset):
    def __init__(self, df_enc, window_size=5):
        self.df = df_enc
        self.window_size = window_size  

    def __len__(self):
        """ Length of the dataset """
        ### BEGIN SOLUTION
        L = len(self.df) - self.window_size + 1

        ### END SOLUTION
        return L

    def __getitem__(self, idx):
        """ returns x[idx], y[idx] for this dataset
        
        x[idx] should be a numpy array of shape (5,)
        """
        ### BEGIN SOLUTION
        if idx + self.window_size > len(self.df):
            x_vals = self.df.iloc[idx:, 0].tolist() + [len(vocab2index)] * (self.window_size - len(self.df) + idx)
            y_vals = self.df.iloc[idx:, 1].tolist() + [len(label2index)] * (self.window_size - len(self.df) + idx)
        else:
            x_vals = self.df.iloc[idx:idx + self.window_size, 0].tolist()
            y_vals = self.df.iloc[idx:idx + self.window_size, 1].tolist()
        
        x = np.array(x_vals, dtype=np.float32)
        central_index = idx + self.window_size // 2
        y = self.df.iloc[central_index]['label']  
        y = np.array(y, dtype=np.int64) 
        
        ### END SOLUTION
        return x, y 


def label_encoding(cat_arr):
   """ Given a numpy array of strings returns a dictionary with label encodings.

   First take the array of unique values and sort them (as strings). 
   """
   ### BEGIN SOLUTION
    # Convert all elements to strings to ensure consistent data type
   cat_arr_str = cat_arr.astype(str)
   unique_vals = np.unique(cat_arr_str)
   sorted_vals = np.sort(unique_vals)
   vocab2index = {val: idx for idx, val in enumerate(sorted_vals)}
   # print("Label to index mapping:", vocab2index)  # Debugging print statement

   ### END SOLUTION
   return vocab2index


def dataset_encoding(df, vocab2index, label2index):
    """Apply vocab2index to the word column and label2index to the label column

    Replace columns "word" and "label" with the corresponding encoding.
    If a word is not in the vocabulary give it the index V=(len(vocab2index))
    """
    V = len(vocab2index)
    df_enc = df.copy()
    ### BEGIN SOLUTION
    df_enc['word'] = df_enc['word'].apply(lambda x: vocab2index.get(x, V))
    df_enc['label'] = df_enc['label'].apply(lambda x: label2index[x])
    ### END SOLUTION
    return df_enc


class NERModel(nn.Module):
    def __init__(self, vocab_size, n_class, emb_size=50, seed=3):
        """Initialize an embedding layer and a linear layer
        """
        super(NERModel, self).__init__()
        torch.manual_seed(seed)
        ### BEGIN SOLUTION
        self.emb_size = emb_size  # Store emb_size as an attribute
        self.embedding = nn.Embedding(vocab_size + 1, emb_size)  # +1 for unknown word
        self.linear = nn.Linear(emb_size * 5, n_class)
        ### END SOLUTION
        
    def forward(self, x):
        """Apply the model to x
        
        1. x is a (N,5). Lookup embeddings for x
        2. reshape the embeddings (or concatenate) such that x is N, 5*emb_size 
           .flatten works
        3. Apply a linear layer
        """
        ### BEGIN SOLUTION
        x = self.embedding(x)  # x shape: (N, 5, emb_size)
        x = x.view(-1, 5*self.emb_size)  # x shape: (N, 5*emb_size)
        x = self.linear(x)  # x shape: (N, n_class)
        ### END SOLUTION
        return x

def get_optimizer(model, lr = 0.01, wd = 0.0):
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    return optim

def train_model(model, optimizer, train_dl, valid_dl, epochs=10):
    for i in range(epochs):
        ### BEGIN SOLUTION
        model.train()  # Set model to training mode
        total_loss = 0
        for x, y in train_dl:
            x = x.long()  # Ensure input tensor is of type LongTensor
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(x)  # Forward pass
            loss = nn.CrossEntropyLoss()(outputs, y)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights
            total_loss += loss.item()
        train_loss = total_loss / len(train_dl)
        ### END SOLUTION
        valid_loss, valid_acc = valid_metrics(model, valid_dl)
        print("train loss  %.3f val loss %.3f and accuracy %.3f" % (
            train_loss, valid_loss, valid_acc))

def valid_metrics(model, valid_dl):
    ### BEGIN SOLUTION
    model.eval()  # Set model to evaluation mode
    criterion = nn.CrossEntropyLoss() 
    total_loss = 0
    correct_predictions = 0
    total_count = 0
    with torch.no_grad():  # No need to track gradients for validation
        for x, y in valid_dl:
            x = x.long()  # Convert x to LongTensor
            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == y).sum().item()
            total_count += y.size(0)
    val_loss = total_loss / len(valid_dl)
    val_acc = correct_predictions / total_count
    ### END SOLUTION
    return val_loss, val_acc

