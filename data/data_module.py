from pytorch_lighting import LightingDataModule


class Recipe_Dataset(LightingDataModule):
    # data files
    PP_RECIPES_FILE = 'PP_recipes.csv'
    INGR_MAP_FILE = 'ingr_map.pkl'

    def __init__(self):
        pass

    # preparation
    def prepare_data(self):
        recipe_df = self.prepare_recipe_dataframe()
        return recipe_df

    def prepare_recipe_dataframe(self):
        recipe_df = pd.read_csv(self.PP_RECIPES_FILE)
        recipe_df.drop([
            'techniques', 'calorie_level', 'steps_tokens', 'name_tokens', 'i',
            'ingredient_tokens'
        ])
        return recipe_df

    def setup(self):
        pass

    # dataloaders
    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass
