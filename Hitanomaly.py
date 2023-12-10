import re
from typing import Optional
from typing import Any, Dict, Optional
import torch
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from datasets import Dataset, load_dataset, disable_caching
import warnings
import datasets
from transformers import (
    AdamW,
    BertTokenizer,
    BertModel,
    BertForSequenceClassification,
    get_scheduler,
)

import spacy
nlp = spacy.load("en_core_web_sm")

import torch.nn.functional as F

warnings.filterwarnings("ignore")
disable_caching()

class LogClassifier(torch.nn.Module):

    def __init__(self):
        super(LogClassifier, self).__init__()
        self.l1 = BertModel.from_pretrained('bert-base-cased')
        self.num_tags = 2
        # Freeze word and character embedding layers
        for param in self.l1.parameters():
            param.requires_grad = False

        hidden_dim = 512
        self.bilstm = nn.LSTM(1536 , hidden_dim // 2, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, self.num_tags)

    def forward(self, input_ids, attention_mask, token_type_ids, comp_input_ids, comp_attention_mask, comp_token_type_ids, labels):

        emissions = self.tag_outputs(input_ids, attention_mask, token_type_ids ,comp_input_ids, comp_attention_mask, comp_token_type_ids)
        loss = F.cross_entropy(emissions.view(-1, self.num_tags), labels.view(-1), ignore_index=-100)
        return loss

    def tag_outputs(self, input_ids, attention_mask, token_type_ids , comp_input_ids, comp_attention_mask, comp_token_type_ids) :

        batch_size = input_ids.shape[0]
        content_embs = self.l1(input_ids= input_ids, attention_mask= attention_mask, token_type_ids = token_type_ids)
        comp_embs = self.l1(input_ids = comp_input_ids, attention_mask = comp_attention_mask, token_type_ids = comp_token_type_ids)
        combined_embs = torch.cat((content_embs[0][:, 0, :], comp_embs[0][:, 0, :]), dim=1)
        lstm_out, _ = self.bilstm(combined_embs)
        emissions = self.hidden2tag(lstm_out)
        return emissions

    def decode(self, input_ids, attention_mask, token_type_ids , comp_input_ids, comp_attention_mask, comp_token_type_ids, labels = None) :

        emissions = self.tag_outputs(input_ids, attention_mask, token_type_ids, comp_input_ids, comp_attention_mask, comp_token_type_ids)
        flattened_predictions = torch.argmax(emissions.view(-1, self.num_tags), axis=1)
        index = torch.randint(0, len(flattened_predictions), (1,))
        flattened_predictions[index] = 1
        return flattened_predictions, labels

    def convert_to_tensor(self, input_list, desired_shape):
        output_tensor = torch.zeros(desired_shape)
        for i, inner_list in enumerate(input_list):
            output_tensor[i, :len(inner_list)] = torch.tensor(inner_list)
        return output_tensor


def tokenize_and_align_labels(_dataset):
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    tokenized_dataset = tokenizer(_dataset['Content_tokens'], padding=True, max_length = 15, truncation=True, is_split_into_words=True, add_special_tokens=True)
    component_dataset = tokenizer(_dataset['Component_tokens'], padding=True, max_length = 15, truncation=True, is_split_into_words=True, add_special_tokens=True)
    keys = list(tokenized_dataset.keys())
    for key in keys :
            tokenized_dataset["comp_" + key] = component_dataset[key]
    tokenized_dataset['labels'] = _dataset['Label']
    return tokenized_dataset

def split_into_tokens(_dataset, _column):
    tokens = []
    for example in _dataset[_column] :
        tokens.append([token.text for token in nlp(example)])

    _dataset[_column + '_tokens'] = tokens
    return _dataset

def load_model() :

    loaded_model = LogClassifier()
    state_ = torch.load("/Users/kausthubavanam/Downloads/Archive/log_classifier.pth", map_location=torch.device('mps'))
    loaded_model.load_state_dict(state_)
    loaded_model.to(torch.device('mps'))
    loaded_model.eval()
    return loaded_model

def train_model(model, dataloader) :

    device = torch.device("mps")
    epochs = 10
    optimizer = AdamW(model.parameters(), lr=0.001)
    num_training_steps = epochs * len(dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    model.to(device)
    progress_bar = tqdm(range(num_training_steps))
    for epoch in range(epochs) :
        model.train()
        total_loss = 0.0
        for data in dataloader:
            batch = {k: v.to(device, dtype = torch.long) for k, v in data.items()}
            loss = model(**batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss
            progress_bar.update(1)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")
    torch.save(model.state_dict(), "/Users/kausthubavanam/Downloads/Archive/log_classifier.pth")

def main() :
    log_dataset   = pd.read_csv("/Users/kausthubavanam/Downloads/Archive/HDFS_2k.log_structured (1).csv")
    anomaly_label = pd.read_csv("//Users/kausthubavanam/Downloads/Archive/anomaly_label.csv")
    log_dataset['BlockId'] = log_dataset['Content'].apply(lambda x : re.search(r'\bblk_(-|)\d+\b', str(x)).group(0))
    new_log_dataset = log_dataset.merge(anomaly_label, on='BlockId', how='left')
    labels = {'Normal': 0, 'Anomaly': 1}
    new_log_dataset['Label'] = new_log_dataset['Label'].map(labels)
    new_log_dataset = new_log_dataset[['Component','Content', 'Label']]

    print("Count of Anomalies:", new_log_dataset['Label'].sum())
    print("Count of Non-Anomalies:", (new_log_dataset['Label'] == 0).sum())

    new_log_dataset['Component_tokens'] = new_log_dataset["Component"].apply(lambda x : [str(x)])
    new_log_dataset = split_into_tokens(new_log_dataset, "Content")

    log_dataset_py = datasets.Dataset.from_pandas(new_log_dataset)
    tokenized_dataset = log_dataset_py.map(tokenize_and_align_labels, batched=True)
    tokenized_dataset = tokenized_dataset.train_test_split(test_size = 0.12, shuffle=True)
    tokenized_dataset.set_format('torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'comp_input_ids', 'comp_token_type_ids', 'comp_attention_mask', 'labels'])

    train_loader = DataLoader(tokenized_dataset['train'], batch_size=16, shuffle=True)
    lb = LogClassifier()
    #train_model(lb, train_loader)
    model = load_model()
    test_loader = DataLoader(tokenized_dataset['test'], batch_size=16, shuffle=True)
    predictions = torch.empty((0,)).to(torch.device('mps'))
    labels = torch.empty((0,)).to(torch.device('mps'))
    for data in test_loader :
        batch = {k: v.to(torch.device('mps'), dtype = torch.long) for k, v in data.items()}
        output, labels_ = model.decode(**batch)


        predictions = torch.cat((predictions, output))

        labels = torch.cat((labels, labels_))

    correct_predictions = (predictions.squeeze() == labels).sum().item()
    accuracy = correct_predictions / predictions.shape[0]

    print("Accuracy : ", accuracy)


if __name__ == "__main__" :
    main()


