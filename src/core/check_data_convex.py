import numpy as np

class DatasetConvexityChecker:

    CONCAVITY_TOLERANCE = 0.80  # Percentage of curvature points that must be <= 0 for "Mostly Concave"
    CONVEXITY_TOLERANCE = 0.80  # Percentage of curvature points that must be >= 0 for "Mostly Convex"

    def __init__(self, prices, demands):
        
        self.prices = np.array(prices)
        self.demands = np.array(demands)

    def check_curvature(self):
        """
        Checks the convexity of the Price vs. Revenue relationship.
        """
        # 1. Calculate Revenue
        revenue = self.prices * self.demands

        # 2. Sort data by Price (keep all points, even with duplicate prices)
        idx = np.argsort(self.prices)
        p = self.prices[idx]
        r = revenue[idx]

        # 3. Calculate First Derivative (Slopes)
        dx = np.diff(p)
        dy = np.diff(r)
        
        # Filter out points where dx is zero to avoid division by zero or NaN in slopes
        non_zero_dx_mask = dx != 0
        if not np.any(non_zero_dx_mask): # All dx are zero, meaning all prices are the same
            return {"curvature": "Cannot calculate curvature due to insufficient distinct price changes"}

        slopes = dy[non_zero_dx_mask] / dx[non_zero_dx_mask]

        # Need at least two slopes to calculate curvature (second derivative)
        if len(slopes) < 2:
            return {"curvature": "Insufficient data for curvature analysis (need at least 2 slopes)"}

        # 4. Calculate Second Derivative (Curvature)
        curvature = np.diff(slopes)

        # 5. Determine Shape based on Curvature (2nd Derivative)
        total_curvature_points = len(curvature)

        if total_curvature_points == 0:
             return {"curvature": "Insufficient data for curvature analysis (no curvature points)"}

        num_concave_points = np.sum(curvature <= 0)
        num_convex_points = np.sum(curvature >= 0)

        percentage_concave = num_concave_points / total_curvature_points
        percentage_convex = num_convex_points / total_curvature_points

        if np.all(curvature <= 0):
            return {"curvature": "Concave (Perfect Hill Shape) - Maximization"}
        elif np.all(curvature >= 0):
            return {"curvature": "Convex (Perfect Bowl Shape) - Minimization"}
        elif percentage_concave >= self.CONCAVITY_TOLERANCE:
            return {"curvature": f"Mostly Concave ({percentage_concave:.2%} points <= 0)"}
        elif percentage_convex >= self.CONVEXITY_TOLERANCE:
            return {"curvature": f"Mostly Convex ({percentage_convex:.2%} points >= 0)"}
        else:
            return {"curvature": "Non-Convex (Mixed/Bumpy)"}