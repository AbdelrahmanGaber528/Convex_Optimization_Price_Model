import numpy as np

class DatasetConvexityChecker:

    def __init__(self, prices, demands):
        
        self.prices = np.array(prices)
        self.demands = np.array(demands)

    def check_curvature(self):
        """
        Checks the convexity of the Price vs. Revenue relationship.
        """
        # 1. Calculate Revenue
        revenue = self.prices * self.demands

        # 2. Sort data by Price 
        idx = np.argsort(self.prices)
        p = self.prices[idx]
        r = revenue[idx] # Use Revenue (r) instead of Demand (d)

        # 3. Calculate First Derivative (Slopes)
        dx = np.diff(p)
        dy = np.diff(r)
        
        # Avoid division by zero if there are duplicate prices
        with np.errstate(divide='ignore', invalid='ignore'):
            slopes = dy / dx

        # 4. Calculate Second Derivative (Curvature)
        curvature = np.diff(slopes)

        # 5. Determine Shape based on Curvature (2nd Derivative)
        if np.all(curvature >= 0):
            return {"curvature": "Convex (Bowl Shape) - Minimization"}
        elif np.all(curvature <= 0):
            return {"curvature": "Concave (Hill Shape) - Maximization"}
        else:
            return {"curvature": "Non-Convex (Mixed/Bumpy)"}