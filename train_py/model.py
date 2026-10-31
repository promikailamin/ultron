#!/usr/bin/env python3
"""PyTorch mirror of src/Ultron.java.

The parameter order and shapes are kept deliberately identical to the Java
implementation so train_py/export.py can write a checkpoint the jar loads
without any conversion code beyond transposing nn.Linear weights.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, width, heads, ffn):
        super().__init__()
        self.ln1 = nn.LayerNorm(width, eps=1e-5)
        self.q = nn.Linear(width, width, bias=True)
        self.k = nn.Linear(width, width, bias=True)
        self.v = nn.Linear(width, width, bias=True)
        self.o = nn.Linear(width, width, bias=True)
        self.ln2 = nn.LayerNorm(width, eps=1e-5)
        self.w1 = nn.Linear(width, ffn, bias=True)
        self.w2 = nn.Linear(ffn, width, bias=True)
        self.heads = heads
        self.head_dim = width // heads

    def forward(self, x, causal_mask):
        b, t, d = x.shape
        h = self.ln1(x)
        q = self.q(h).view(b, t, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(h).view(b, t, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(h).view(b, t, self.heads, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(causal_mask[:, :, :t, :t] == 0, float("-inf"))
        att = torch.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(b, t, d)
        x = x + self.o(y)
        h = self.ln2(x)
        x = x + self.w2(F.gelu(self.w1(h), approximate="tanh"))
        return x


class UltronModel(nn.Module):
    def __init__(self, vocab, width=384, layers=8, heads=6, ffn=1536, ctx=512):
        super().__init__()
        assert width % heads == 0, "width must be divisible by heads"
        self.vocab = vocab
        self.width = width
        self.layers = layers
        self.heads = heads
        self.ffn = ffn
        self.ctx = ctx
        self.wte = nn.Embedding(vocab, width)
        self.wpe = nn.Embedding(ctx, width)
        self.blocks = nn.ModuleList([Block(width, heads, ffn) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(width, eps=1e-5)
        self.apply(self._init)
        mask = torch.tril(torch.ones(ctx, ctx, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, ctx, ctx), persistent=False)

    @staticmethod
    def _init(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, mask=None):
        b, t = idx.shape
        if t > self.ctx:
            raise ValueError("sequence length %d exceeds context %d" % (t, self.ctx))
        pos = torch.arange(t, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)[None, :, :]
        for block in self.blocks:
            x = block(x, self.causal_mask)
        x = self.ln_f(x)
        logits = x @ self.wte.weight.t()
        if targets is None:
            return logits
        if mask is None:
            loss = F.cross_entropy(logits.view(-1, self.vocab), targets.view(-1))
        else:
            loss = F.cross_entropy(logits.view(-1, self.vocab), targets.view(-1),
                                   reduction="none")
            loss = (loss * mask.view(-1)).sum() / mask.sum().clamp(min=1)
        return logits, loss

    def config(self):
        return {
            "vocab": self.vocab,
            "width": self.width,
            "layers": self.layers,
            "heads": self.heads,
            "ffn": self.ffn,
            "ctx": self.ctx,
        }


def build_model(config):
    return UltronModel(
        vocab=config["vocab"],
        width=config["width"],
        layers=config["layers"],
        heads=config["heads"],
        ffn=config["ffn"],
        ctx=config["ctx"],
    )
