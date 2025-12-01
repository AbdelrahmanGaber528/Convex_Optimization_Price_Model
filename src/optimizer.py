from src import ALPHA , BETA , MAX_PRICE , MAX_DEMAND_CAPACITY
import cvxpy as cp 


class PricingModel:

    def __init__(self):

        self.alpha = ALPHA 
        self.beta = BETA
        self.max_demand_capacity = MAX_DEMAND_CAPACITY
        self.max_price = MAX_PRICE
        self.constraints = []

    def _solve_convex(self, price , demand ):

        self.constraints.append()


        
