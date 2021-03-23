import torch
import torch.nn as nn
import pytorch_lightning as pl
from omegaconf import DictConfig
import numpy as np
from time import time

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
        # flags
        self.file = config.file.log
        # build data module
        self.dataset = RecipeDataModule(config)
        self.dataset.setup()
        # layers
        self.embedding = nn.Embedding(
            self.dataset.vocab_size, config.model.d_model
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
            config.model.d_model, self.dataset.vocab_size
        )
        self.softmax = nn.Softmax(dim=(-1))
        # loss
        # self.loss = nn.CrossEntropyLoss()

    def forward(
            self, X: torch.Tensor, mask: torch.Tensor, pad_mask: torch.Tensor
        ):
        ''' '''
        E = self.embedding(X)
        E = self.encoder(E, mask=mask, src_key_padding_mask=pad_mask)
        logits = self.decoder(E)
        preds = self.softmax(logits)
        return preds

    # def loss(self, ):

    # configuration
    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(),
            lr=self.lr,
            # betas=(self.beta,) # TODO: implement
        )

    # masking
    def augment_batch(self, batch: torch.Tensor):
        '''
        Applies masking to batch.
        Args:
            batch:  [b x seqlen]
        '''
        batch_shape = batch.size()
        # ---------------------- one way --------------------
        # IDEA: so we could do it this way where every sequence has ~15% masked
        # or we give every token a 15% probability of being masked. The former
        # is more conceptually favorable to me, but other way is easier and
        # appears in conventional implementations.
        # # [b] | get number of indecies to mask per sequence in batch
        # seqlen = torch.where(batch!=self.dataset.PAD_ID, True, False).sum(1)
        # # [b] | calc number of tokens to augment per sequence
        # n_augmented = (self.frac_augmented * seqlen).round()
        # print(f'n_augmented: {n_augmented}')
        # ---------------------- other way --------------------
        s = time()
        # [b x seqlen] | all tokens tagged for augmentation
        is_augmented = (
            (torch.rand(batch_shape) < self.frac_augmented)
            & (batch != self.dataset.PAD_ID)
        )
        # [b x seqlen] | select values to mask
        is_masked = (
            (torch.rand(batch_shape) < self.frac_masked) & is_augmented
        )
        # [b x seqlen] | select values to randomize
        # WARNING: this is not yet the proper math to maintain actual ratios
        is_random = (
            (torch.rand(batch_shape) < self.frac_random)
            & is_augmented
            & ~is_masked
        )
        # [] | total number of random tokens to generate
        n_random = is_random.sum()
        print(f'n_random: {n_random}')

        # mask batch
        # batch = batch.where(is_masked, self.dataset.MASK_ID)
        batch[is_masked] = self.dataset.MASK_ID
        # randomize batch
        print(f'random_tokens: {self.random_tokens(is_random.sum())}')
        batch[is_random] =  self.random_tokens(is_random.sum())
        print(f'random: {batch}')
        print(f'time: {time() - s}')
        raise RuntimeError()
        return batch

    def random_tokens(self, n: int):
        ''' Generates n random tokens in usable vocab '''
        x = np.random.choice(
            range(self.dataset.vocab_size), size=n, replace=True
        )
        raise RuntimeError(x)
        return x

    # steps
    def training_step(self, batch: torch.Tensor, batch_idx: int):
        ''' '''
        # apply augmentations
        augmented_batch = self.augment_batch(batch)
        # get predictions
        preds = self(augmented_batch)
        # calculate loss
        # loss = self.loss(preds, batch)
        # # log
        # if self.file:
        #     pass
        # return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        pass

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        pass

    # data loaders
    def train_dataloader(self):
        return self.dataset.train_dataloader()

    def val_dataloader(self):
        return self.dataset.val_dataloader()

    def test_dataloader(self):
        return self.dataset.test_dataloader()
