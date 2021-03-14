import pickle
import pandas as pd
from cached_property import cached_property
from pytorch_lightning import LightningDataModule


class Recipe_Dataset(LightningDataModule):
    # data files
    DATA_ROOT_DIR = 'data/archive'
    PP_RECIPES_FILE = f'{DATA_ROOT_DIR}/PP_recipes.csv'
    INGR_MAP_FILE = f'{DATA_ROOT_DIR}/ingr_map.pkl'

    def __init__(self):
        super().__init__()

    def __repr__(self):
        return 'Recipe_Dataset()'

    # preparation
    def prepare_data(self):
        recipe_df = self.prepare_recipe_dataframe()
        ingr_map = self.prepare_ingr_map()
        return recipe_df

    def prepare_recipe_dataframe(self):
        recipe_df = pd.read_csv(self.PP_RECIPES_FILE)
        recipe_df = recipe_df.drop(
            columns=[
                'techniques', 'calorie_level', 'steps_tokens',
                'name_tokens', 'i', 'ingredient_tokens'
            ]
        )
        return recipe_df

    def prepare_ingr_map(self):
        with open(self.INGR_MAP_FILE) as ingr_map_file:
            ingr_map = pickle.load(ingr_map_file)
        return ingr_map

    def setup(self):
        pass

    # dataloaders
    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass
