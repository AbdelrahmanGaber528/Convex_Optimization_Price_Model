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
        self.min_price = self.cost_per_unit / (1 - self.min_profit_margin)
        
        self.price = cp.Variable(pos=True, name="Price")

        self.demand = self.alpha - self.beta * self.price
        self.problem = None

        self.constraints = [
            self.demand <= self.max_capacity,
            self.price <= self.max_price,
            self.price >= self.min_price,
            self.demand >= 0
        ]

        self.current_model_type = None

    def _build_concave_model(self):
        """Build the standard Model"""
        revenue = ( self.alpha * self.price ) - ( self.beta * cp.power(self.price, 2) )
        objective = cp.Maximize(revenue)
        self.problem = cp.Problem(objective, self.constraints)
        self.current_model_type = "concave"
        return revenue

    def solve_concave(self):
        """Solve the concave model (maximize revenue)"""
        try:
            self.problem.solve()
            final_demand = self.alpha - self.beta * self.price.value
            return [self.price.value, self.problem.value, final_demand, self.problem.status]
        except Exception as e:
            print(f"Error solving concave model: {e}")
            return [None, None, None, "Failed"]

    def check_convexity_curve_dcp(self):
        """
        Checks if the objective function is Concave and if the problem is DCP compliant.
        Returns: dict with curvature info and DCP status.
        """
        revenue = self.alpha * self.price - self.beta * cp.power(self.price, 2)
        curvature = revenue.curvature
        is_dcp_compliant = self.problem.is_dcp() if self.problem else False

        info = {
            'curvature': str(curvature),
            'justification': "",
            'conclusion': "",
            'is_dcp_compliant': is_dcp_compliant,
            'dcp_conclusion': "The problem is DCP compliant, meaning it can be solved by convex optimization solvers." if is_dcp_compliant else "The problem is NOT DCP compliant, which may lead to incorrect or suboptimal solutions with convex optimization solvers."
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
        """
        hessian = -2 * self.beta
        
        if hessian < 0:
            result = "CONCAVE"
        elif hessian > 0:
            result = "CONVEX"
        else:
            result = "LINEAR"
        
        return {
            'method': 'Hessian Matrix (Second Derivative)',
            'hessian_value': hessian,
            'second_derivative': str(hessian),
            'result': result,
            'is_concave': hessian < 0,
            'is_convex': hessian > 0,
            'suitable_for_maximization': hessian < 0
        }

    def build_convex_model(self):
        """
        Convert to CONVEX model by negating objective
        """    
        g_of_R = -self.alpha * self.price + self.beta * cp.power(self.price, 2)
        self.problem = cp.Problem(cp.Minimize(g_of_R), self.constraints)
        self.current_model_type = "convex"
        return g_of_R

    def solve_convex(self):
        """Solve the convex model (minimize g(R))"""
        try:
            self.problem.solve()
            final_demand = self.alpha - self.beta * self.price.value
            return [self.price.value, self.problem.value, final_demand, self.problem.status]
        except Exception as e:
            print(f"Error solving convex model: {e}")
            return [None, None, None, "Failed"]

    def build_nonconvex_model(self):
        """
        Construct a NON-CONVEX and NON-CONCAVE revenue model.
        """
        base_revenue = self.alpha * self.price - self.beta * cp.power(self.price, 2)
        cubic_term = 0.001 * cp.power(self.price, 3)
        revenue = base_revenue + cubic_term
        objective = cp.Maximize(revenue)
        self.problem = cp.Problem(objective, self.constraints)
        self.current_model_type = "nonconvex_nonconcave"
        return revenue

    def restore_convex_model(self):
        """Restores the model to its convex version."""
        self.build_convex_model()

    def calculate_concave_revenue(self, prices_array):
        """
        Concave revenue: R(p) = α*p - β*p^2
        """
        return prices_array * (self.alpha - self.beta * prices_array)

    def calculate_convex_revenue(self, prices_array):
        """
        Convex transformation of revenue for CVXPY solver: g(p) = -R(p)
        """
        return -prices_array * (self.alpha - self.beta * prices_array)

    def calculate_nonconvex_revenue(self, prices_array):
        """
        Non-convex, non-concave revenue with a cubic term.
        """
        return prices_array * (self.alpha - self.beta * prices_array) + 0.001 * np.power(prices_array, 3)

