import re
import torch
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cached_property import cached_property
from pytorch_lightning import LightningDataModule


class RecipeDataModule(LightningDataModule):
    ADDITIONAL_TOKENS = [
        'MASK', # tokens for model to predict
        'UNK',  # default for unknown tokens
        'PAD',   # token for padding in sequence
    ]

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
    def vocab_size(self):
        ''' Number of unique ingredients in dataset '''
        return len(self.ingr_id_to_name_map)

    @cached_property
    def num_recipes(self):
        ''' Number of unique recipes in dataset '''
        return self.recipe_df.count(axis='columns')

    # converters
    @cached_property
    def ingr_id_to_name_map(self):
        if not hasattr(self, 'ingr_map'):
            raise RuntimeError(
                'Must load ingr_map before calling ingr_id_to_name_map.'
            )
        return {
            row['id'] : row['replaced'] for _, row in self.ingr_map.iterrows()
        }

    @cached_property
    def ingr_name_to_id_map(self):
        if not hasattr(self, 'ingr_map'):
            raise RuntimeError(
                'Must load ingr_map before calling ingr_name_to_id_map.'
            )
        return {
            row['replaced'] : row['id'] for _, row in self.ingr_map.iterrows()
        }

    def food_id_to_name(self, id):
        return self.ingr_id_to_name_map[id]

    def food_name_to_id(self, name):
        return self.ingr_name_to_id_map[id]

    @classmethod
    def dataframe_to_tensor(cls, dataframe):
        ingredient_data = dataframe['ingredient_ids'].map(
            cls.id_string_to_tensor
        )
        ingredient_data = torch.nn.utils.rnn.pad_sequence(
            ingredient_data,  batch_first=True
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

    # preparation
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
        ####
        # train_data = recipe_dataset.sample(frac=self.train_frac)
        # remaining_data = recipe_dataset.drop(train_data.index)
        # val_count = int(self.val_frac * len(recipe_dataset))
        # val_data = remaining_data.sample(n=val_count)
        # test_data = remaining_data.drop(val_data.index)
        # # reset indecies to be 0 through len indexeds
        # train_data = train_data.reset_index()
        # val_data = val_data.reset_index()
        # test_data = test_data.reset_index()
        # # cast as torch datasets
        # train_data = RecipeDataset(train_data)
        # val_data = RecipeDataset(val_data)
        # test_data = RecipeDataset(test_data)
        # return (train_data, val_data, test_data)

    def prepare_ingr_map(self):
        with open(self.INGR_MAP_PATH, 'rb') as INGR_MAP_PATH:
            ingr_map = pickle.load(INGR_MAP_PATH)
        # update with additional tokens
        for i, x in enumerate(sorted(list(set(ingr_map['id'])))):
            assert (i==x), (i,x)
        return ingr_map

    # setup
    def prepare_data(self):
        pass

    def setup(self):
        # load files
        recipe_dataset = self.prepare_recipe_dataset()
        self.ingr_map = self.prepare_ingr_map()
        # split and cache datasets
        (
            self.train_data,
            self.val_data,
            self.test_data
        ) = self.split_data(recipe_dataset)

    # dataloaders
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_data, batch_size=self.batch_size
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_data, batch_size=self.batch_size
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_data, batch_size=self.batch_size
        )
