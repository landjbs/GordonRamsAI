import torch
import torch.nn as nn
import pytorch_lightning as pl


class Model(pl.LightningModule):
    ''' '''

    def __init__(self, config, vocab_size: int):
        super().__init__()
        # dims
        d_model = config.model.d_model
        # layers
        self.embedding = nn.Embedding(
            vocab_size, d_model
        )
        self.transformer = nn.Transformer(
            d_model=d_model,
            dim_feedforward=(2*d_model),
            nhead=config.model.nhead,
            num_encoder_layers=config.model.num_encoder_layers,
            num_decoder_layers=config.model.num_decoder_layers
        )
        self.decoder = nn.Linear(d_model, vocab_size)
        self.softmax = nn.Softmax(dim=(-1))

    def forward(self, X: torch.Tensor):
        ''' '''
        E = self.embedding(X)
        E = self.transformer(E)
        logits = self.decoder(E)
        pi = self.softmax(logits)
        return logits

    # def init_params(self):
        # for n, p in self.named_parameters():
            # nn.init.xavier_uniform_(p)
