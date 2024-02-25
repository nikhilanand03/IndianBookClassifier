import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

print(os.path)
model = torch.load('/Users/nikhilanand/JupyterNotebooks/Sarvam.ai/roberta model/model_save.pth')

tokenizer = AutoTokenizer.from_pretrained('roberta-base')

def construct_prompt(title, author, series, lang):
    prompt = "The title of the book is " + str(title) + ", and the author is " + str(author) + ". The book is from the " + str(series) + " series.\
 The book is written in " + str(lang) + ". Is the book's context Indian or Non-Indian?"

    return prompt

def get_args(title,author,series,lang):
    text = construct_prompt(title,author,series,lang)
        
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    return {
        'input_ids': encoding['input_ids'].flatten(),
        'attention_mask': encoding['attention_mask'].flatten()
    }

title = input("Enter title: ")
author = input("Enter author: ")
series = input("Enter series (NA=not applicable): ")
lang = input("Enter language: ")

args = get_args(title,author,series,lang)
args['input_ids'] = args['input_ids'].unsqueeze(0)
args['attention_mask']=args['attention_mask'].unsqueeze(0)

output = model(input_ids=args['input_ids'],attention_mask=args['attention_mask']).logits[0]
label = np.argmax(nn.Softmax(dim=0)(output).detach().numpy())
out_label = "Indian" if label==1 else "Non-Indian"
print(out_label)