import re
import torch
import pickle
import numpy as np
import pandas as pd
from cached_property import cached_property
from pytorch_lightning import LightningDataModule


def id_strings_to_id_tensor(s):
    ''' Converts stirng of ids to tensor '''
    int_splits = map(int, re.findall('\d+(?=[,\]])', s))
    data = torch.tensor(int_splits)
    print(data)
    return data


class RecipeDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe):
        super().__init__()
        self.length = len(dataframe)
        self.data = dataframe

    def __len__(self):
        return self.length

    def __getitem__(self, key):
        # grab row at index
        series = self.data.loc[key]
        # cast data segment as tensor
        return torch.tensor(series)


class RecipeDataModule(LightningDataModule):
    # data files
    DATA_ROOT_DIR = 'data/archive'
    PP_RECIPES_FILE = f'{DATA_ROOT_DIR}/PP_recipes.csv'
    INGR_MAP_FILE = f'{DATA_ROOT_DIR}/ingr_map.pkl'

    def __init__(self, config):
        super().__init__()
        self.batch_size = config.batch_size
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

    def split_data(self, recipe_dataset):
        ''' split training data '''
        #
        train_data = recipe_dataset.sample(frac=self.train_frac)
        remaining_data = recipe_dataset.drop(train_data.index)
        val_count = int(self.val_frac * len(recipe_dataset))
        val_data = remaining_data.sample(n=val_count)
        test_data = remaining_data.drop(val_data.index)
        # reset indecies to be 0 through len indexeds
        train_data = train_data.reset_index()
        val_data = val_data.reset_index()
        test_data = test_data.reset_index()
        # cast as torch datasets
        train_data = RecipeDataset(train_data)
        val_data = RecipeDataset(val_data)
        test_data = RecipeDataset(test_data)
        # for data in [train_data, val_data, test_data]:
            # series = data['ingredient_ids'].map(lambda x: len(x))
            # print(f'min: {series.min()} max: {series.max()}')
        return (train_data, val_data, test_data)

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
        pass

    def test_dataloader(self):
        pass

    # helpers
    @staticmethod
    def id_strings_to_id_tensor(s):
        ''' Converts stirng of ids to tensor '''
        return map(int, re.findall('\d+(?=[,\]])', s))
