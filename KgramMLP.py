# starter code by matus & o1-pro
import argparse
import csv
import os
import time
import random
import math
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
################################################################################
# 3. K-gram MLP in a sequence-to-sequence approach
################################################################################


class KGramMLPSeqModel(nn.Module):
    def __init__(self, vocab_size, k=3, embed_size=512, num_inner_layers=1, hidden_dim=None, chunk_size = 1):
        super().__init__()
        self.k = k
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_inner_layers = num_inner_layers
        self.chunk_size = chunk_size
        self.use_cache = False # this is for kv cache which should not be applicable here.

        self.embedding = nn.Embedding(vocab_size, embed_size)


        if hidden_dim is None:
            hidden_dim = embed_size // 2


        layers = [nn.Linear(k * embed_size, hidden_dim), nn.GELU()]
        for _ in range(num_inner_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        layers.append(nn.Linear(hidden_dim, vocab_size))

        self.net = nn.Sequential(*layers)

        if hidden_dim is None:
            hidden_dim = embed_size // 2


        layers = [nn.Linear(k * embed_size, hidden_dim), nn.GELU()]
        for _ in range(num_inner_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        layers.append(nn.Linear(hidden_dim, vocab_size))

        self.net = nn.Sequential(*layers)

    def forward(self, tokens_seq, use_cache=False):
        """
        tokens_seq: (seq_len, batch)
        Return: (seq_len, batch, vocab_size)
        """
        seq_len, batch_size = tokens_seq.shape
        device = tokens_seq.device
 
        pad = torch.zeros(self.k - 1, batch_size, dtype=torch.long, device=device)
        padded = torch.cat([pad, tokens_seq], dim=0)  
 
        context_windows = []
        for i in range(self.k):
            context_windows.append(padded[i:i + seq_len])
        contexts = torch.stack(context_windows, dim=2)   
        embedded = self.embedding(contexts)  
        flat = embedded.reshape(seq_len, batch_size, self.k * self.embed_size)

        
        
        chunks = torch.split(flat, self.chunk_size, dim=0)
        
        logit_chunks = []
        for chunk in chunks:
            logit_chunk = self.net(chunk) 
            logit_chunks.append(logit_chunk)
            
        logits = torch.cat(logit_chunks, dim=0)
         
        return logits


