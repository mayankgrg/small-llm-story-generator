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
class TransformerModel(nn.Module):
    def __init__(self, vocab_size=50257, d_model=1024, n_heads=2, n_blocks=4, mlp_g_hidden_layer_output=2048, max_seq_len=1024, use_kv_cache=False, use_rope=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList([TransformerBlock(vocab_size, d_model, n_heads, mlp_g_hidden_layer_output, use_rope) for i in range(n_blocks)])
        self.unembedding = nn.Linear(d_model, vocab_size)
        self.kv_cache = [None] * n_blocks
        self.use_cache = use_kv_cache
        self.use_rope = use_rope

        pass

    def forward(self, tokens_seq, use_cache=False):
        x = self.embedding(tokens_seq)

        if use_cache and self.kv_cache[0] is not None:
            pass
        else:
            if use_cache:
                self.kv_cache = [None] * len(self.blocks)

        for i in range(len(self.blocks)):
            kv_cache_layer_i = self.kv_cache[i] if use_cache else None
            x, new_kv_cache = self.blocks[i](x, kv_cache_layer_i)
            self.kv_cache[i] = new_kv_cache

        logits = self.unembedding(x)

        return logits

    def reset_cache(self):
        self.kv_cache = [None] * len(self.blocks)
