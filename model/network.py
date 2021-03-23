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
        # self.beta = config.optimizer.beta
        # masking
        self.frac_augmented = config.masking.frac_augmented
        self.frac_masked = config.masking.frac_masked
        self.frac_random = config.masking.frac_random
        self.frac_unchanged = config.masking.frac_unchanged
        # build data module
        self.data_module = RecipeDataModule(config)
        self.data_module.setup()
        # layers
        self.embedding = nn.Embedding(
            self.data_module.vocab_size, config.model.d_model
        )
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
        self.decoder = nn.Linear(
            config.model.d_model, self.data_module.vocab_size
        )
        self.softmax = nn.Softmax(dim=(-1))
        # loss
        self.loss = nn.CrossEntropyLoss()

    def forward(self, X: torch.Tensor):
        ''' '''
        E = self.embedding(X)
        E = self.encoder(E)
        logits = self.decoder(E)
        preds = self.softmax(logits)
        return preds

    # configuration
    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(),
            lr=self.lr,
            # betas=(self.beta,) # TODO: implement
        )

    # masking
    def augment_batch(self, batch: torch.Tensor):
        ''' Applies masking to batch '''
        pass

    # steps
    def training_step(self, batch: torch.Tensor, batch_idx: int):
        ''' '''
        # apply augmentations
        augmented_batch = self.augment_batch(batch)
        # get predictions


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
