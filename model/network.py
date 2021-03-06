import torch
import torch.nn as nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from omegaconf import DictConfig
import numpy as np
from time import time
from termcolor import colored

from data import RecipeDataModule


class Model(pl.LightningModule):
    '''
    '''

    def __init__(self, config: DictConfig):
        super().__init__()
        # hparams
        self.lr = config.optimizer.lr
        self.config = config
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
            num_embeddings=self.dataset.token_count,
            embedding_dim=config.model.d_model,
            padding_idx=self.dataset.PAD_ID
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
        self.ignore_index = -1
        self.cross_entropy = nn.CrossEntropyLoss(
            weight=self.soft_token_weights(),
            ignore_index=self.ignore_index
        )

    def soft_token_weights(self, eps: float = 1.) -> torch.Tensor:
        weights = (1. / torch.log(self.dataset.token_frequencies + 1))
        # plt.plot(sorted(weights))
        # plt.show()
        # raise RuntimeError()
        return weights

    def visualize_sentence(
        self, ground_truth: torch.Tensor, targets: torch.Tensor,
        preds: torch.Tensor = None, topk: int = 3
        ) -> str:
        ''' '''
        sentence = ''
        top_preds = ''

        for true_id, target_id, pred in zip(ground_truth, targets, preds):
            if true_id == self.dataset.PAD_ID:
                continue
            true_name = self.dataset.food_id_to_name(true_id)
            if target_id==self.ignore_index:
                is_augmented = False
            else:
                is_augmented = True
                pred_ps, pred_ids = pred.topk(k=topk)
                top_preds += f'{true_name}:\n\t' # colored( , "blue")
                top_preds += '\n\t'.join(
                    ( # colored
                        # (
                        f'{(pred_name := self.dataset.food_id_to_name(pred_id))}'
                        f' {round(float(pred_p), 5)}'
                        # ),
                        # color='green' if pred_name==true_name else 'red'
                    )
                    for pred_p, pred_id in zip(pred_ps, pred_ids)
                )
                top_preds += '\n'
            if is_augmented:
                true_name = true_name.upper()
            sentence += f'{true_name} | ' # colored(, color=token_color)

        str_out = f'{sentence}\n{top_preds}'
        return str_out

    def visualize(
        self, ground_truth: torch.Tensor, targets: torch.Tensor,
        preds: torch.Tensor = None, topk: int = 3
        ):
        str_out = ''
        for i in range(min(ground_truth.size(0), 5)):
            str_out += (
                f'{self.visualize_sentence(ground_truth[i], targets[i], preds[i])}'
                f'\n{"-"*80}\n'
            )
        return str_out

    def forward(self, X: torch.Tensor, pad_mask: torch.Tensor):
        '''
        Forward pass through the network.
        Args:
            X:          Padded input sequence [b x n x d_m]
            pad_mask:   Binary tensor denoting masked tokens with True [b x n]
        Returns:
            preds:      Normalized prediction distribution for each token
                        [b x n x vocab_size]
        '''
        # [b x n] -> [b x n x d_m] | fetch embedding vectors of each id
        E = self.embedding(X)
        # [b x n x d_m] -> [n x b x d_m] |
        E = self.encoder(E.transpose(1, 0), src_key_padding_mask=pad_mask)
        # [n x b x d_m] -> [b x n x vocab_size] |
        logits = self.decoder(E.transpose(1, 0))
        # [b x n x vocab_size] -> [b x n x vocab_size] |
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
        '''
        Applies masking to batch.
        Args:
            batch:  [b x n]
        '''
        batch_shape = batch.size()
        # ---------------------- one way --------------------
        # IDEA: so we could do it this way where every sequence has ~15% masked
        # or we give every token a 15% probability of being masked. The former
        # is more conceptually favorable to me, but other way is easier and
        # appears in conventional implementations.
        # # [b] | get number of indecies to mask per sequence in batch
        # n = torch.where(batch!=self.dataset.PAD_ID, True, False).sum(1)
        # # [b] | calc number of tokens to augment per sequence
        # n_augmented = (self.frac_augmented * n).round()
        # print(f'n_augmented: {n_augmented}')
        # ---------------------- other way --------------------
        # [b x n] | init target sequence for prediction
        targets = batch.clone()
        # [b x n] | mask for all padded tokens
        pad_mask = (batch == self.dataset.PAD_ID)
        # [b x n] | all tokens tagged for augmentation
        is_augmented = (
            (torch.rand(batch_shape) < self.frac_augmented)
            & ~pad_mask
        )
        # [b x n] | select values to mask
        is_masked = (
            (torch.rand(batch_shape) < self.frac_masked) & is_augmented
        )
        # [b x n] | select values to randomize
        # WARNING: this is not yet the proper stats to maintain actual ratios
        is_random = (
            (torch.rand(batch_shape) < self.frac_random)
            & is_augmented
            & ~is_masked
        )
        # mask batch
        batch[is_masked] = self.dataset.MASK_ID
        # randomize batch
        batch[is_random] = self.random_tokens(int(is_random.sum()))
        # ignore all non-augmented indecies of targets
        targets[~is_augmented] = self.ignore_index
        return batch, targets, pad_mask

    def random_tokens(self, n: int) -> torch.Tensor:
        ''' Generates n random tokens in usable vocab '''
        return torch.randint(high=self.dataset.vocab_size, size=(n,))

    # steps
    def training_step(self, batch: torch.Tensor, batch_idx: int):
        ''' '''
        # apply augmentations
        masked_batch, targets, pad_mask = self.augment_batch(batch.clone())
        # [b x n x vocab_size] | get predictions
        preds = self(masked_batch, pad_mask)
        # [(b * n) x v], [(b * n)] | calculate loss
        loss = self.cross_entropy(preds.flatten(0, 1), targets.flatten())
        # log
        if self.file:
            self.log('Train/Loss', loss.detach())
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        if self.file:
            # apply augmentations
            masked_batch, targets, pad_mask = self.augment_batch(batch.clone())
            # [b x n x vocab_size] | get predictions
            preds = self(masked_batch, pad_mask)
            # [(b * n) x v], [(b * n)] | calculate loss
            loss = self.cross_entropy(preds.flatten(0, 1), targets.flatten())
            # log metrics
            self.log('Validation/Loss', loss.detach())
            # analyze
            # display some completions
            self.logger.experiment.add_text(
                'Validation/Completions',
                self.visualize(batch, targets, preds),
                global_step=self.global_step
            )
            # display distribution over tokens
            self.logger.experiment.add_histogram(
                'Validation/Distribution',
                preds.flatten(0, 1).mean(),
                global_step=self.global_step
            )

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        pass

    # setup
    def on_train_start(self):
        if self.file:
            self.logger.experiment.add_text('Config', self.config.pretty())

    # data loaders
    def train_dataloader(self):
        return self.dataset.train_dataloader()

    def val_dataloader(self):
        return self.dataset.val_dataloader()

    def test_dataloader(self):
        return self.dataset.test_dataloader()
