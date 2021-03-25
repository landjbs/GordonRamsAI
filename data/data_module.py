from typing import Union

import re
import torch
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cached_property import cached_property
from pytorch_lightning import LightningDataModule
from termcolor import colored


class RecipeDataModule(LightningDataModule):
    # NOTE: ids of special tokens can be accessed with {name}_ID (eg. CLS_ID).
    CLS_TOKEN = 'CLS' # token for start of a sequence
    SEP_TOKEN = 'SEP' # token for seperating two sequences
    MASK_TOKEN = 'MASK' # tokens for model to predict
    UNK_TOKEN = 'UNK' # default for unknown tokens
    PAD_TOKEN = 'PAD'  # token for padding in sequence
    SPECIAL_TOKENS = [CLS_TOKEN, SEP_TOKEN, MASK_TOKEN, UNK_TOKEN, PAD_TOKEN]

    def __init__(self, config):
        super().__init__()
        # datasets
        self.RECIPIES_PATH = config.datasets.recipes_path
        self.INGR_MAP_PATH = config.datasets.ingr_map_path
        # train
        self.batch_size = config.train.batch_size
        self.train_frac = config.train.train_frac
        self.val_frac = config.train.val_frac
        self.test_frac = config.train.test_frac
        self.num_workers = config.compute.num_workers

    def __repr__(self):
        repr_data = ['batch_size']
        s = 'Recipe_Dataset('
        for i, name in enumerate(repr_data):
            if (i != 0):
                s += ', '
            s += f'{name}={getattr(self, name)}'
        return s + ')'

    # specs
    @cached_property
    def token_count(self):
        '''
        Number of unique tokens. Distinct from vocab_size in that it includes
        special tokens in addition to ingredients.
        '''
        return len(self.ingr_name_to_id_map)

    @cached_property
    def vocab_size(self):
        ''' Number of unique ingredients in dataset '''
        return (self.token_count - len(self.SPECIAL_TOKENS))

    @cached_property
    def num_recipes(self):
        ''' Number of unique recipes in dataset '''
        return self.recipe_df.count(axis='columns')

    # converters
    @cached_property
    def ingr_id_to_name_map(self):
        return {id : name for name, id in self.ingr_name_to_id_map.items()}

    @cached_property
    def ingr_name_to_id_map(self):
        if not hasattr(self, 'ingr_map'):
            raise RuntimeError(
                'Must load ingr_map before calling ingr_name_to_id_map.'
            )
        # build ingr_name_to_id_map from ingr_map dataframe
        ingr_name_to_id_map = {
            row['replaced'] : row['id'] for _, row in self.ingr_map.iterrows()
        }
        # get ids for special tokens
        special_start = self.ingr_map['id'].max() + 1
        special_end = special_start + len(self.SPECIAL_TOKENS)
        special_ids = range(special_start, special_end)
        # add special tokens to map
        for id, token in zip(special_ids, self.SPECIAL_TOKENS):
            # TODO: this is horrible naming technique that depends on the tokens
            # value not its name in the class. Must be changed asap.
            name = f'{token.split("_")[0]}_ID'
            setattr(self, name, id)
            ingr_name_to_id_map.update({token: id})
        return ingr_name_to_id_map


    def food_id_to_name(self, id: Union[int, torch.Tensor]):
        return self.ingr_id_to_name_map[int(id)]

    def food_name_to_id(self, name: str):
        return self.ingr_name_to_id_map[id]

    def dataframe_to_tensor(self, dataframe):
        ingredient_data = dataframe['ingredient_ids'].map(
            self.id_string_to_tensor
        )
        ingredient_data = torch.nn.utils.rnn.pad_sequence(
            ingredient_data,  batch_first=True, padding_value=self.PAD_ID
        )
        return ingredient_data

    @staticmethod
    def id_string_to_tensor(s: str):
        ''' Converts string of ids to tensor. '''
        # convert string to int list
        ids = list(map(int, re.findall('\d+(?=[,\]])', s)))
        # cast int list as tensor
        ids = torch.tensor(ids) # , dtype=torch.int16
        return ids

    def translate(self, tensor: torch.Tensor) -> str:
        ''' Translates tensor into string of ingredients '''
        # TODO: allow different dims of tensor
        # OPTIMIZE: this obviously
        f = lambda x : str(self.food_id_to_name(int(x)))
        if ((t_dim := tensor.dim()) == 1):
            return ', '.join(map(f, tensor))
        elif (t_dim == 2):
            raise NotImplementedError()
        else:
            raise ValueError('Invalid dims for translated tensor.')

    def visualize(
        self, ground_truth: torch.Tensor, augmented: torch.Tensor,
        targets: torch.Tensor, preds: torch.Tensor
        ) -> str:
        ''' Visualizes '''
        
        return s

    # preparation
    def mask_ingredient_tensor(self, data):
        ''' '''
        return data
        # select indecies for augmentation in each sequence
        # determine and apply type of augmentation
        # add cls tokens to the start of each sequence
        # data =

    def prepare_recipe_dataset(self):
        # load dataframe
        recipe_df = pd.read_csv(self.RECIPIES_PATH)
        # remove unnecessary columns
        recipe_df = recipe_df.drop(
            columns=[
                'techniques', 'calorie_level', 'steps_tokens',
                'name_tokens', 'i', 'ingredient_tokens'
            ]
        )
        # convert ingredient_ids to tensor
        recipe_data = self.dataframe_to_tensor(recipe_df)
        # apply masking scheme to recipe_data
        # recipe_data = self.mask_ingredient_tensor(recipe_data)
        return recipe_data

    def split_data(self, recipe_dataset: torch.Tensor):
        ''' Split training data into train/val/test tensors '''
        # determine sizes for each set
        n_train = int(self.train_frac * (n := recipe_dataset.size(0)))
        n_val = int(self.val_frac * n)
        n_test = int(n - (n_train + n_val))
        # split data
        train_data, val_data, test_data = torch.utils.data.random_split(
            recipe_dataset, lengths=[n_train, n_val, n_test]
        )
        return train_data, val_data, test_data

    def prepare_ingr_map(self) -> pd.DataFrame:
        # load ingr_map
        with open(self.INGR_MAP_PATH, 'rb') as INGR_MAP_PATH:
            ingr_map = pickle.load(INGR_MAP_PATH)
        # save ingr_map
        self.ingr_map = ingr_map
        # update with additional tokens and cache map
        _ = self.ingr_name_to_id_map
        return True

    # setup
    def prepare_data(self):
        pass

    def setup(self):
        # load files
        self.ingr_map = self.prepare_ingr_map()
        recipe_dataset = self.prepare_recipe_dataset()
        # split and cache datasets
        (
            self.train_data,
            self.val_data,
            self.test_data
        ) = self.split_data(recipe_dataset)

    # dataloaders
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_data, batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_data, batch_size=self.batch_size
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_data, batch_size=self.batch_size
        )
