from data import RecipeDataModule
from config import Config

config = Config()
x = RecipeDataModule(config)
x.prepare_ingr_map()

# for y in x.test_dataloader():
    # print(y)
