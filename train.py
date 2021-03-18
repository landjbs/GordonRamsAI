from data import RecipeDataModule
from config import Config

config = Config()
x = RecipeDataModule(config)
x.setup()
for y in x.train_dataloader():
    print(y)
