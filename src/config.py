from dataclasses import dataclass

@dataclass
class Params:
    ALPHA: int = 1500
    BETA: int = 2
    MAX_PRICE: float = 600.0
    MAX_CAPACITY: int = 1200
    MIN_PROFIT_MARGIN: float = 0.20
    COST_PER_UNIT: float = 100

## Some Explanation :

#  alpha : is market size where , if price is (p) , how many customers can pay to it ?

# beta : is price sensitivity , if price is (p)  , how many customers we will lose ? 

# max_price : the maximum we can put to price ( Based on law or other thing )

# max_capacity : the max amount of product you can sell . factory limit