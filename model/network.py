import torch
import torch.nn as nn
import pytorch_lightning as pl
from omegaconf import DictConfig

from data import RecipeDataModule


class Model(pl.LightningModule):
    ''' '''

    def __init__(self, config: DictConfig):
        super().__init__()
        # hparams
        self.lr = config.optimizer.lr
        self.beta = config.optimizer.beta
        # build data module
        self.data_module = RecipeDataModule(config)
        # layers
        self.embedding = nn.Embedding(vocab_size, config.model.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.model.d_model,
            nhead=config.model.nhead,
            dim_feedforward=(2 * config.model.d_model),
            dropout=config.model.dropout,
            activation=config.model.activation
        )
        self.encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=config.model.num_layers,
                # norm=
        )
        self.decoder = nn.Linear(config.model.d_model, vocab_size)
        self.softmax = nn.Softmax(dim=(-1))

    def forward(self, X: torch.Tensor):
        ''' '''
        E = self.embedding(X)
        E = self.encoder(E)
        logits = self.decoder(E)
        pi = self.softmax(logits)
        return logits

    # configuration
    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(),
            lr=self.lr,
            # betas=(self.beta,) # TODO: implement
        )

    # steps
    def training_step(self, batch: torch.Tensor, batch_idx: int):
        print(self(batch).shape)

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        pass

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        pass

    # data loaders
    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        return self.data_module.test_dataloader()
