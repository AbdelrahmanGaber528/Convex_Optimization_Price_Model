import numpy as np
import pandas as pd

def load_sample_dataset():

    prices = np.array([5, 10, 15, 20, 25, 30, 35])
    demands = np.array([115, 105, 92, 70, 50, 30, 10])
    min_price = prices.min()
    max_price = prices.max()
    return prices, demands, min_price, max_price

def load_from_csv(path):

    df = pd.read_csv(path)
    prices = df['price'].values
    demands = df['demand'].values
    min_price = prices.min()
    max_price = prices.max()
    return prices, demands, min_price, max_price