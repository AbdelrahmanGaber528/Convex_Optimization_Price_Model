import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
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

    def check_dataset_convexity_convex_hull(self):
        """
        Check if dataset is convex using convex hull.
        Updates hull points and hull revenue.
        """

        try:
            hull = ConvexHull(self.dataset)
            self.hull_points = self.dataset[hull.vertices]
            
            sorted_hull = self.hull_points[np.argsort(self.hull_points[:,0])]
            self.hull_revenue = np.interp(self.prices, sorted_hull[:,0], sorted_hull[:,1])
            
            is_convex = len(hull.vertices) == len(self.dataset)
            
            self.curvature = "CONVEX" if is_convex else "NOT CONVEX"
            
            print(f"   Total points: {len(self.dataset)}")
            print(f"   Hull vertices: {len(hull.vertices)}")
            print(f"   Hull area: {hull.volume:.2f}")
            
            if is_convex:
                print("\nDataset is CONVEX (all points on hull)")
            else:
                print("\nDataset is NOT CONVEX")
            
            return {
                'hull': hull,
                'is_convex': is_convex,
                'vertices': hull.vertices,
                'n_vertices': len(hull.vertices),
                'n_points': len(self.dataset)
            }
            
        except Exception as e:
            print(f"Convex hull computation failed: {e}")
            return None


    def make_convex(self):
        """
        Transform the dataset into a convex one by projecting revenues
        onto the convex envelope.
        """

        sorted_idx = np.argsort(self.prices)
        x_sorted = self.prices[sorted_idx]
        y_sorted = self.revenue[sorted_idx]

        hull = ConvexHull(np.column_stack((x_sorted, y_sorted)))
        hull_points = np.column_stack((x_sorted, y_sorted))[hull.vertices]
        hull_points = hull_points[np.argsort(hull_points[:,0])]

        y_convex = np.interp(x_sorted, hull_points[:,0], hull_points[:,1])

        self.revenue = y_convex
        self.dataset = np.column_stack((self.prices, self.revenue))

        print(" Dataset transformed to convex shape using convex envelope.")

        self.check_dataset_convexity_convex_hull()


    def plot_hull(self, save_path=None):
        """
        Plot dataset with convex hull and highlight points above/below hull.
        Optionally save plot to file.
        """
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
            print(f"Plot saved to {save_path}")

        plt.close()
