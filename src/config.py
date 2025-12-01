from dataclasses import dataclass

@dataclass
class Params:
    ALPHA: int = 1000
    BETA: float = 2.0
    MAX_PRICE: float = 150.0
    MAX_DEMAND_CAPACITY: int = 2500 

## Some Explanation : 

#  alpha : is market size where , if price is (p) , how many customers can pay to it ?

# beta : is price sensitivity , if price is (p)  , how many customers we will lose ? 

# max_price : the maximum we can put to price ( Based on law or other thing )

# max_demand_capacity : the max amount of product you can sell . 