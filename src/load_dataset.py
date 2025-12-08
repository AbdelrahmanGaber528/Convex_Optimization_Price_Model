import numpy as np
import pandas as pd

def load_sample_dataset():

    prices = np.array([5, 10, 15, 20, 25, 30, 35])
    demands = np.array([115, 105, 92, 70, 50, 30, 10])
    return prices, demands

def load_from_csv(path):

    df = pd.read_csv(path)
    return df['price'].values, df['demand'].values