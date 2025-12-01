import cvxpy as cp
import numpy as np
from .config import Params as ConfigParams 

class PricingModel:

    def __init__(self):
        self.alpha = ConfigParams.ALPHA 
        self.beta = ConfigParams.BETA
        self.max_capacity = ConfigParams.MAX_CAPACITY
        self.max_price = ConfigParams.MAX_PRICE

        self.price = cp.Variable(pos=True, name="Price")
        self.demand = None
        self.problem = None

    def _build_convex_model(self):
        """Build the standard Quadratic Model"""

        # Equations 
        self.demand = self.alpha - self.beta * self.price
        
        revenue = self.alpha * self.price - self.beta * cp.power(self.price, 2)
        
        constraints = [
            self.demand <= self.max_capacity,
            self.price <= self.max_price,   
            self.demand >= 0 
        ]

        # Define the problem
        self.problem = cp.Problem(cp.Maximize(revenue), constraints)

    def solve_convex(self):
        """Solves the standard convex model"""

        if self.problem is None:
            print("Building Convex Model...")
            self._build_convex_model()

        try:
            self.problem.solve()
            print(f"Convex Problem Solved Successfully")
            
            final_demand = self.alpha - self.beta * self.price.value
            
            return [self.price.value, self.problem.value, final_demand, self.problem.status]
            
        except cp.error.SolverError:
            print("Solver Failed.")
            return [None, None, None, "Failed"]

    def check_convexity(self):
        """
        Checks if the objective function is Concave (valid for Maximization).
        Returns: dict with curvature info.
        """
        revenue = self.alpha * self.price - self.beta * cp.power(self.price, 2)
        curvature = revenue.curvature
        
        info = {
            'curvature': str(curvature),
            'justification': "",
            'conclusion': ""
        }

        if curvature == 'CONCAVE':
            info['justification'] = f"The second derivative is negative (${-2 * self.beta}$). This indicates a concave function."
            info['conclusion'] = "The function is **Concave**, guaranteeing a global maximum, making it suitable for maximization using convex optimization."
        elif curvature == 'CONVEX':
            info['justification'] = f"The second derivative is positive (${2 * self.beta}$"
            info['conclusion'] = "The function is **Convex**, meaning a global minimum is guaranteed. It is **not suitable for maximization** without modification or seeking a minimum."
            
        return info


    def _build_nonconvex_model(self):
        """
        Build a non-convex model with an added sinusoidal term to the revenue.
        The non-convexity is introduced by a term like 7000 * sin(price).
        """
        self.demand = self.alpha - self.beta * self.price
        
        convex_revenue = self.alpha * self.price - self.beta * cp.power(self.price, 2)

        # Non-convex term: 7000 * price^3 (Maximizing a convex function is non-DCP)
        non_convex_revenue_term = 7000 * cp.power(self.price, 3) # This makes it non-convex

        # Total revenue
        revenue = convex_revenue + non_convex_revenue_term
        
        constraints = [
            self.demand <= self.max_capacity,
            self.price <= self.max_price,   
            self.demand >= 0 
        ]

        self.problem = cp.Problem(cp.Maximize(revenue), constraints)

        if not self.problem.is_dcp():
            print("\n" + "="*80)
            print("The non-convex model is NOT DCP compliant.")
            print("Attempting to solve this problem with a DCP-only solver will raise an error.")


    def restore_convex_model(self):
        """Restores the model to its original convex form."""

        self._build_convex_model()
        print("Convex Model Restored.")


    def calculate_convex_revenue(self, prices_array):
        """Calculates numerical revenue for convex model given an array of prices."""
        
        demand = self.alpha - self.beta * prices_array
        demand = np.maximum(demand, 0)
        return prices_array * demand

    def calculate_non_convex_revenue(self, prices_array):
        """Calculates numerical revenue for non-convex model given an array of prices."""

        convex_revenue = self.calculate_convex_revenue(prices_array)
        non_linear_term = 7000 * np.sin(prices_array)
        return convex_revenue + non_linear_term