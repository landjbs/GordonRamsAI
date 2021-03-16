import re
import torch
import pickle
import numpy as np
import pandas as pd
from cached_property import cached_property
from pytorch_lightning import LightningDataModule


class Recipe_Dataset(LightningDataModule):
    # data files
    DATA_ROOT_DIR = 'data/archive'
    PP_RECIPES_FILE = f'{DATA_ROOT_DIR}/PP_recipes.csv'
    INGR_MAP_FILE = f'{DATA_ROOT_DIR}/ingr_map.pkl'

    def __init__(self, config):
        super().__init__()
        self.train_frac = config.train_frac
        self.val_frac = config.val_frac
        self.test_frac = config.test_frac

    def __repr__(self):
        return 'Recipe_Dataset()'

    # specs
    @cached_property
    def num_ingredients(self):
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

    # preparation
    def prepare_recipe_dataset(self):
        # load dataframe
        recipe_df = pd.read_csv(self.PP_RECIPES_FILE)
        # remove unnecessary columns
        recipe_df = recipe_df.drop(
            columns=[
                'techniques', 'calorie_level', 'steps_tokens',
                'name_tokens', 'i', 'ingredient_tokens'
            ]
        )
        return recipe_df

    def prepare_ingr_map(self):
        with open(self.INGR_MAP_FILE, 'rb') as ingr_map_file:
            ingr_map = pickle.load(ingr_map_file)
        return ingr_map

    # setup
    def prepare_data(self):
        pass

    def setup(self):
        # load files
        recipe_dataset = self.prepare_recipe_dataset()
        ingr_map = self.prepare_ingr_map()
        # split training data
        train_data = recipe_dataset.sample(frac=self.train_frac)
        remaining_data = recipe_dataset.drop(train_data.index)
        val_data = remaining_data.sample(frac=self.val_frac)
        test_data = remaining_data.drop(val_data.index)
        # cache datasets
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        print(self.train_data.head())
        print(self.val_data.head())
        print(self.test_data.head())
        # return recipe_dataset

    # dataloaders
    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass
