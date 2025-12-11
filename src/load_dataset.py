import numpy as np
import pandas as pd
from src.config import Params

def load_dataset():

    np.random.seed(42) 

    # Generate prices
    prices = np.linspace(5, 35, 200)

    demand_base = Params.ALPHA - Params.BETA * prices

    demands = demand_base 
  
    demands[demands < 0] = 0

    return prices, demands
