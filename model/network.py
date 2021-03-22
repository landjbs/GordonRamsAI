import torch
import torch.nn as nn
import pytorch_lightning as pl


class Model(pl.LightningModule):
    def __init__(self, config, vocab_size: int):
        self.embedding = nn.Embedding(
            vocab_size, config.model.d_model
        )
        self.transformer = nn.Transformer(
            d_model=config.model.d_model,
            n_heads=config.model.n_heads
            dim_feedforward=(2*config.model.d_model)
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
