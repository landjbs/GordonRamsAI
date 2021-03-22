import torch
import torch.nn as nn
import pytorch_lightning as pl
from omegaconf import DictConfig


class Model(pl.LightningModule):
    ''' '''

    def __init__(self, config: DictConfig, vocab_size: int):
        super().__init__()
        # hparams
        self.lr = config.optimizer.lr
        self.beta = config.optimizer.beta
        # dims
        d_model = config.model.d_model
        # layers
        self.embedding = nn.Embedding(
            vocab_size, d_model
        )
        # self.transformer = nn.Transformer(
        #     d_model=d_model,
        #     dim_feedforward=(2*d_model),
        #     nhead=config.model.nhead,
        #     num_encoder_layers=config.model.num_encoder_layers,
        #     num_decoder_layers=config.model.num_decoder_layers
        # )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config.model.nhead,
            dim_feedforward=(2*d_model),
            dropout=config.model.dropout,
            activation=config.model.activation
        )

        self.decoder = nn.Linear(d_model, vocab_size)
        self.softmax = nn.Softmax(dim=(-1))

    # configuration
    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(),
            lr=self.lr,
            # betas=(self.beta,) # TODO: implement
        )

    #
    def forward(self, X: torch.Tensor):
        ''' '''
        E = self.embedding(X)
        E = self.transformer(E)
        logits = self.decoder(E)
        pi = self.softmax(logits)
        return logits

    # steps
    def training_step(self, batch: torch.Tensor, batch_idx: int):
        self(batch)

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        pass

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        pass
