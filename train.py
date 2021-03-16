from data import Recipe_Dataset
from config import Config

config = Config()
x = Recipe_Dataset(config)
x.setup()
