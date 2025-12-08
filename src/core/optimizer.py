import cvxpy as cp
import sympy as sp
import numpy as np
from ..config import Params as ConfigParams

class PricingModel:

    def __init__(self):

        self.alpha = ConfigParams.ALPHA 
        self.beta = ConfigParams.BETA
        self.max_capacity = ConfigParams.MAX_CAPACITY
        self.max_price = ConfigParams.MAX_PRICE
        self.cost_per_unit = ConfigParams.COST_PER_UNIT
        self.min_profit_margin = ConfigParams.MIN_PROFIT_MARGIN
        self.min_price = self.cost_per_unit  / (1 - self.min_profit_margin)

        self.price = cp.Variable(pos=True, name="Price")
        self.demand = None
        self.problem = None


    def _build_model(self):
        """Build the standard Model"""

        # Equations 
        self.demand = self.alpha - self.beta * self.price
        
        revenue = ( self.alpha * self.price ) - ( self.beta * cp.power(self.price, 2) )
        
        constraints = [
            self.demand <= self.max_capacity,
            self.price <= self.max_price,   
            self.price >= self.min_price,
            self.demand >= 0
        ]

        # Define the problem
        self.problem = cp.Problem(cp.Maximize(revenue), constraints)


    def solve_convex(self):
        """Solves the problem"""

        if self.problem is None:
            print("Building Model...")
            self._build_model()

        try:
            self.problem.solve()
            
            final_demand = self.alpha - self.beta * self.price.value
            print(f"Convex Problem Solved Successfully")
            return [self.price.value, self.problem.value, final_demand, self.problem.status]

        except cp.error.SolverError:
            print("Solver Failed.")
            print(f"Optimization failed with status: {self.problem.status}")
            return [None, None, None, "Failed"]


    def check_convexity_cvxpy(self):
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
            info['justification'] = f"The second derivative is negative (${-2*self.beta}$). This indicates a concave function."
            info['conclusion'] = "The function is **Concave**, guaranteeing a global maximum, making it suitable for maximization using convex optimization."
            
        elif curvature == 'CONVEX':
            info['justification'] = f"The second derivative is positive (${2*self.beta}$). This implies that 'beta' is negative. For price sensitivity, 'beta' is typically positive, which would result in a concave revenue function."
            info['conclusion'] = "The function is **Convex**, meaning a global minimum is guaranteed. It is **not suitable for maximization** without modification or seeking a minimum."
            
        return info


    def check_convexity_hessian(self):
        """
        Check convexity using Hessian Matrix (Second Derivative)
        
        Mathematical Analysis:
        ---------------------
        Revenue Function: R(p) = α*p - β*p²
        
        First Derivative (slope):
        dR/dp = α - 2β*p
        
        Second Derivative (Hessian - curvature):
        d²R/dp² = -2β
        
        Interpretation:
        - If Hessian < 0: CONCAVE (curves down, mountain shape) → Good for MAX
        - If Hessian > 0: CONVEX (curves up, bowl shape) → Good for MIN
        - If Hessian = 0: LINEAR (no curvature)
        """
        
        hessian = -2 * self.beta
        
        print("HESSIAN MATRIX ANALYSIS")
        print(f"Revenue Function: R(p) = {self.alpha}p - {self.beta}p²")
        
        if hessian < 0:
            print(f"Result: CONCAVE (Hessian = {hessian} < 0)")
            print("   Shape: Mountain (curves downward)")
            print("   Optimization: Perfect for MAXIMIZATION")
            print("   Guarantee: Global maximum exists and can be found")
            result = "CONCAVE"

        elif hessian > 0:
            print(f"Result: CONVEX (Hessian = {hessian} > 0)")
            print("   Shape: Bowl (curves upward)")
            print("   Optimization: Perfect for MINIMIZATION")
            print("   Problem: Not suitable for maximization")
            result = "CONVEX"
        
        return {
            'method': 'Hessian Matrix (Second Derivative)',
            'hessian_value': hessian,
            'second_derivative': str(hessian),
            'result': result,
            'is_concave': hessian < 0,
            'is_convex': hessian > 0,
            'suitable_for_maximization': hessian < 0
        }


    def check_convexity(self):

        # 1. Define Symbols
        p_sym = sp.Symbol('p') 
        a_sym = sp.Symbol('alpha')
        b_sym = sp.Symbol('beta') 


        revenue_func = a_sym * p_sym - b_sym * p_sym**2

        second_derivative = sp.diff(revenue_func, p_sym, 2)

        hessian_value = float(second_derivative.subs({b_sym: self.beta}))

        # 6. Conclusion
        if hessian_value < 0:
            print("5. Conclusion:\n   Hessian is NEGATIVE. Function is CONCAVE (Hill shape).")
            print("   -> GLOBAL MAXIMUM GUARANTEED.")
            return "CONCAVE"
        
        elif hessian_value > 0:
            print("5. Conclusion:\n   Hessian is POSITIVE. Function is CONVEX (Bowl shape).")
            print("   -> Suitable for Minimization only.")
            return "CONVEX"


    def _build_nonconvex_model(self):
        """
        Build a non-convex model with an added absolute value term to the revenue.
        Non-convexity is introduced by subtracting a term like 7000 * abs(price - 600).
        """
        self.demand = self.alpha - self.beta * self.price

        base_revenue = self.alpha * self.price - self.beta * cp.power(self.price, 2)

        nonconvex_term = 0.001 * cp.power(self.price, 3)
        revenue = base_revenue + nonconvex_term

        constraints = [
            self.demand <= self.max_capacity,
            self.price <= self.max_price,
            self.price >= self.min_price,
            self.demand >= 0
        ]

        self.problem = cp.Problem(cp.Maximize(revenue), constraints)

        print("NON-CONVEX Model Built")

        if not self.problem.is_dcp():
            print("Model is NOT DCP compliant (expected for non-convex)")


    def restore_convex_model(self):
        """Restores the model to its original convex form."""
       
        self._build_model()

        print("Convex Model Restored.")




    def calculate_convex_revenue(self, prices_array):
        """Calculates numerical revenue for convex model given an array of prices."""
        
        demand = self.alpha - self.beta * prices_array
        return prices_array * demand


    def calculate_non_convex_revenue(self, prices_array):
        """
        Calculate numerical revenue for non-convex model with abs term.
        """

        convex_revenue = self.calculate_convex_revenue(prices_array)

        nonconvex_term = 0.001 * np.power(prices_array, 3)

        return convex_revenue + nonconvex_term

