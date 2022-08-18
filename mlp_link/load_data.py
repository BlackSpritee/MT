import json
from transformers import BertTokenizer
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional
from torch.utils.data import Dataset, DataLoader
# from openpyxl import load_workbook
import random
import numpy as np



def load_infer(batch_data):
    max_length=128
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    train_data = {}
    train_data['mask'] = []
    train_data['text'] = []

    for line in batch_data:
        sub,sub_type,obj,obj_type,reviewbody=line["sub"],line["sub_type"],line["obj"],line["obj_type"],line["reviewbody"]
        sentence=reviewbody.strip()
        sent="[unused0]"+str(sub)+"[unused0]"+"[unused1]"+str(sub_type)+"[unused1]"+"[unused2]"+str(obj)+"[unused2]"+"[unused3]"+str(obj_type)+"[unused3]"+"[unused4]"+str(sentence)+"[unused4]"
        indexed_tokens = tokenizer.encode(sent, add_special_tokens=True)
        avai_len = len(indexed_tokens)
        while len(indexed_tokens) <  max_length:
            indexed_tokens.append(0)  # 0 is id for [PAD]
        indexed_tokens = indexed_tokens[: max_length]
        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)  # (1, L)

        # Attention mask
        att_mask = torch.zeros(indexed_tokens.size()).long()  # (1, L)
        att_mask[0, :avai_len] = 1
        train_data['text'].append(indexed_tokens)
        train_data['mask'].append(att_mask)
    train_text = torch.tensor([ t.numpy() for t in train_data['text']])
    train_mask = torch.tensor([ t.numpy() for t in train_data['mask']])
    return train_text,train_mask

