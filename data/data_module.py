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
        with open(self.INGR_MAP_FILE, 'rb') as ingr_map_file:
            ingr_map = pickle.load(ingr_map_file)
        return ingr_map

    # setup
    def prepare_data(self):
        pass

    def setup(self):
        # load files
        recipe_df = self.prepare_recipe_dataframe()
        ingr_map = self.prepare_ingr_map()
        # split training data

        return recipe_df

    # dataloaders
    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass
