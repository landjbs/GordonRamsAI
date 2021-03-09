from pytorch_lighting import LightingDataModule


class Recipe_Dataset(LightingDataModule):
    # data files
    PP_RECIPES_FILE = 'PP_recipes.csv'
    INGR_MAP_FILE = 'ingr_map.pkl'

    def __init__(self):
        pass

    def prepare_data(self):
        pass

    def setup(self):
        pass

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass
