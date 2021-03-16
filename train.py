from data import Recipe_Dataset
from config import Config

config = Config()
x = Recipe_Dataset(config)
x.setup()
for y in x.train_dataloader():
    print(y)
