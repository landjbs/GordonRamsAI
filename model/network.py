import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, config):
        self.embedding = nn.Embedding(
            config.vocab_size, config.d_model
        )
        self.transformer = nn.Transformer(
            d_model=config.d_model,
            n_heads=config.n_heads
            dim_feedforward=(2*config.d_model)
        )

    def forward(self, X: torch.Tensor, M: torch.Tensor):
        E = self.embedding(X)
        E = self.transformer(E)
