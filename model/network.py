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
        self.decoder = nn.Linear(config.d_model, config.vocab_size)
        self.softmax = nn.Softmax(dim=(-1))

    def forward(self, X: torch.Tensor, M: torch.Tensor):
        ''' '''
        E = self.embedding(X)
        E = self.transformer(E)
        logits = self.decoder(E)
        pi = self.softmax(logits)
        return logits

    # def init_params(self):
        # for n, p in self.named_parameters():
            # nn.init.xavier_uniform_(p)
