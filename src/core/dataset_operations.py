import numpy as np
from scipy.spatial import ConvexHull
from scipy.interpolate import interp1d

class Dataset:
    """
    Check convexity/concavity of a Price vs Revenue dataset using Convex Hull,
    with visualization, saving capability, and convex transformation.
    """

    def __init__(self, prices, demands):
        self.prices = np.array(prices)
        self.demands = np.array(demands)
        self.revenue = self.prices * self.demands
        self.dataset = np.column_stack((self.prices, self.revenue))

        self.hull_points = None
        self.hull_revenue = None
        self.curvature = None
        self._previous_revenue_before_nonconvex = None # Store revenue before making non-convex
        self._previous_demands_before_nonconvex = None # Store demands before making non-convex

    def check_dataset_convexity_convex_hull(self):
        """
        Check if dataset is convex using convex hull.
        Stores:
            - self.hull_points
            - self.hull_revenue
            - self.curvature
        """

        try:
            points = self.dataset
            hull = ConvexHull(points)

            # Extract hull points (unsorted)
            hull_points = points[hull.vertices]

            # Sort hull points by price for consistent interpolation
            hull_points = hull_points[np.argsort(hull_points[:, 0])]

            # Save internally
            self.hull_points = hull_points

            # Interpolate convex hull revenue curve across dataset prices
            self.hull_revenue = np.interp(self.prices, hull_points[:, 0], hull_points[:, 1])

            # Check convexity by ensuring all revenue points are on or below the hull
            is_convex = np.all(self.revenue <= self.hull_revenue + 1e-8)

            self.curvature = "CONVEX" if is_convex else "NOT CONVEX"

            return {
                'hull': hull,
                'is_convex': is_convex,
                'vertices': hull.vertices,
                'n_vertices': len(hull.vertices),
                'n_points': len(points)
            }

        except Exception as e:
            print(f"Convex hull computation failed: {e}")
            return None



    def make_convex(self):
        """
        Method: Replace original revenue with the convex hull revenue.
        This restores convexity by pulling non-convex points down to the hull.
        """

        # Ensure hull is computed
        if self.hull_points is None or self.hull_revenue is None:
            self.check_dataset_convexity_convex_hull()

        # Replace original revenue with the convex hull's upper envelope
        self.revenue = self.hull_revenue

        # Update demands to stay consistent with the new revenue
        # Avoid division by zero with a small epsilon
        self.demands = self.revenue / (self.prices + 1e-9)
        self.dataset = np.column_stack((self.prices, self.revenue))

        # Recompute hull after transformation
        self.check_dataset_convexity_convex_hull()

        return {
            "status": "revenue replaced with convex hull",
            "points_updated": len(self.prices)
        }

    def make_nonconvex(self, amplitude=500, frequency_factor=2):
        """
        Transforms the dataset to a non-convex shape by adding a sine wave to the revenue.
        Stores the current (convex) state for later restoration.
        """
        self._previous_revenue_before_nonconvex = self.revenue.copy()
        self._previous_demands_before_nonconvex = self.demands.copy()

        # Add a deterministic non-convex component to the revenue
        self.revenue = self.revenue + amplitude * np.sin(self.prices / frequency_factor)
        
        # Update demands to stay consistent with the new revenue
        self.demands = self.revenue / (self.prices + 1e-9)
        self.dataset = np.column_stack((self.prices, self.revenue))

        # Recompute hull after transformation
        self.check_dataset_convexity_convex_hull()

        return {
            "status": "dataset transformed to non-convex",
            "points_updated": len(self.prices)
        }

    def restore_from_nonconvex(self):
        """
        Restores the dataset to the state it was in before make_nonconvex was called.
        Then, it calls make_convex to ensure it's convex again.
        """
        if self._previous_revenue_before_nonconvex is not None:
            self.revenue = self._previous_revenue_before_nonconvex
            self.demands = self._previous_demands_before_nonconvex
            self.dataset = np.column_stack((self.prices, self.revenue))
            
            # Ensure it's convex after restoration
            self.make_convex()

            self._previous_revenue_before_nonconvex = None
            self._previous_demands_before_nonconvex = None

            return {
                "status": "dataset restored and made convex",
                "points_updated": len(self.prices)
            }
        else:
            return {"status": "No previous non-convex state to restore from."}


    def plot_hull(self, save_path=None):
        """
        Plot dataset with convex hull and highlight points above/below hull.
        Optionally save plot to file.
        """
        import matplotlib.pyplot as plt
        if self.hull_points is None or self.hull_revenue is None:
            self.check_dataset_convexity_convex_hull()

        plt.figure(figsize=(10,6))

        plt.scatter(self.prices, self.revenue, color='blue', label='Dataset Revenue')
        plt.scatter(self.hull_points[:,0], self.hull_points[:,1], color='red', s=80, label='Hull Vertices')
        plt.plot(self.prices, self.hull_revenue, 'r--', linewidth=2, label='Convex Hull Revenue')

        above_mask = self.revenue > self.hull_revenue + 1e-8
        below_mask = self.revenue < self.hull_revenue - 1e-8

        if np.any(above_mask):
            plt.scatter(self.prices[above_mask], self.revenue[above_mask], color='orange', label='Above Hull')
        if np.any(below_mask):
            plt.scatter(self.prices[below_mask], self.revenue[below_mask], color='green', label='Below Hull')

        plt.title(f"Price vs Revenue with Convex Hull ({self.curvature})")
        plt.xlabel("Price")
        plt.ylabel("Revenue")
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)

        plt.close()
