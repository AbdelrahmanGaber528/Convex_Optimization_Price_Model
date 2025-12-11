import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

Y_MAX = 2000 

def plot_dataset_revenue(prices, demands, save_path='reports/dataset_revenue.png'):
    """
    Plots the revenue from the raw dataset (price * demand).
    """
    revenue = prices * demands

    # Sort by price for a continuous line plot
    sorted_indices = prices.argsort()
    sorted_prices = prices[sorted_indices]
    sorted_revenue = revenue[sorted_indices]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(sorted_prices, sorted_revenue, marker='o', linestyle='-', label='Dataset Revenue')
    ax.set_xlabel('Price')
    ax.set_ylabel('Revenue')
    ax.set_title('Price vs. Revenue from Dataset')
    ax.legend()
    _save_plot(fig, save_path)
    print(f'Saved {save_path}')


def plot_convex_hull(prices, demands, save_path='reports/convex_hull_check.png', title='Convex Hull Geometric Check'):
    """
    Plots the raw data points and wraps them in a Convex Hull (Red Polygon).
    """
    revenue = prices * demands
    points = np.column_stack((prices, revenue))

    try:
        hull = ConvexHull(points)
    except:
        print("Could not compute convex hull for plotting.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(prices, revenue, 'o', color='blue', label='Data Points')

    for simplex in hull.simplices:
        ax.plot(points[simplex, 0], points[simplex, 1], 'r-', linewidth=2,
                label='Convex Hull' if 'Convex Hull' not in [l.get_label() for l in ax.get_lines()] else "")

    ax.plot(points[hull.vertices, 0], points[hull.vertices, 1], 'r*', markersize=12, label='Hull Vertices')

    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Price')
    ax.set_ylabel('Revenue')
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_plot(fig, save_path)
    print(f"Saved {save_path}")


def plot_separate_revenues(prices, rev_convex, rev_nonconvex, rev_bowl, optimal_price, optimal_revenue,
                           save_dir='reports'):
    """
    Generates three separate images:
    1. Original Concave Hill
    2. Non-Convex S-Shape
    3. Convex Bowl (New Demo)
    """

    # 1. Plot Original Concave Hill
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(prices, rev_convex, label='Original (Concave)', color='blue', linewidth=2)
    if optimal_price is not None:
        ax1.scatter([optimal_price], [optimal_revenue], color='blue', s=100, label=f'Max Price: ${optimal_price:.2f}')
    ax1.set_title('Original Revenue (Concave Hill)')
    ax1.set_xlabel('Price')
    ax1.set_ylabel('Revenue')
    ax1.set_ylim(bottom=0, top=Y_MAX)  # Ensure y-axis starts at 0 for clarity
    ax1.set_xlim(prices.min(), prices.max())
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    _save_plot(fig1, os.path.join(save_dir, 'concave_revenue.png'))

    # 2. Plot Non-Convex S-Shape
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(prices, rev_nonconvex, label='Non-Convex (S-Shape)', color='orange', linewidth=2, linestyle='--')
    ax2.set_title('Non-Convex Revenue (S-Shape)')
    ax2.set_xlabel('Price')
    ax2.set_ylabel('Revenue')
    ax2.set_ylim(bottom=0, top=Y_MAX)  # Ensure y-axis starts at 0 for clarity
    ax2.set_xlim(prices.min(), prices.max())
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    _save_plot(fig2, os.path.join(save_dir, 'nonconvex_revenue.png'))

    # 3. Plot Convex Bowl (Demo)
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.plot(prices, rev_bowl, label='Transformed (Convex Bowl)', color='green', linewidth=2, linestyle='-.')
    if optimal_price is not None:
        ax3.scatter([optimal_price], [-optimal_revenue], color='green', s=100, label=f'Min Price: ${optimal_price:.2f}')

    ax3.set_title('Transformed Model (Convex Bowl)')
    ax3.set_xlabel('Price')
    ax3.set_ylabel('Negative Revenue')
    ax3.set_xlim(prices.min(), prices.max())
    ax3.set_ylim(bottom=-Y_MAX, top=0) # Negative revenue
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    _save_plot(fig3, os.path.join(save_dir, 'convex_bowl_revenue.png'))


def plot_comparison_all_three(prices, rev_convex, rev_nonconvex, rev_bowl,
                              optimal_price, optimal_revenue, save_path='reports/revenue_comparison_combined.png'):
    fig, ax = plt.subplots(figsize=(12, 8))

    # 1. Concave Hill (Blue)
    ax.plot(prices, rev_convex, label='Original (Concave Hill)', color='blue', linewidth=3)

    # 2. Non-Convex S-Shape (Orange)
    ax.plot(prices, rev_nonconvex, label='Non-Convex (S-Shape)', color='orange', linewidth=3, linestyle='--')

    # 3. Convex Bowl (Green)
    ax.plot(prices, rev_bowl, label='Convex Bowl (Demo)', color='green', linewidth=3, linestyle='-.')

    ax.axhline(0, color='black', linewidth=1, alpha=0.5)

    if optimal_price is not None:
        ax.scatter([optimal_price], [optimal_revenue], color='blue', s=100, zorder=5, label=f'Max Price: ${optimal_price:.2f}')
        ax.scatter([optimal_price], [-optimal_revenue], color='green', s=100, zorder=5, label=f'Min Price: ${optimal_price:.2f}')

    ax.set_xlabel('Price ($)')
    ax.set_ylabel('Revenue ($)')
    ax.set_title('Comparison: Concave Hill vs. S-Shape vs. Convex Bowl')
    ax.set_xlim(prices.min(), prices.max())
    ax.set_ylim(bottom=-Y_MAX, top=Y_MAX) # Y-axis from -Y_MAX to Y_MAX
    ax.legend()
    ax.grid(True, alpha=0.3)

    _save_plot(fig, save_path)
    print(f'Saved {save_path}')


def plot_revenue_curve(prices, revenue, title, save_path, optimal_price=None, optimal_revenue=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(prices, revenue, linewidth=2)
    if optimal_price is not None and optimal_revenue is not None:
        ax.scatter([optimal_price], [optimal_revenue], s=100, zorder=5)
    ax.set_title(title)
    ax.set_xlabel('Price')
    ax.set_ylabel('Revenue')
    ax.set_ylim(bottom=0, top=Y_MAX) # Ensure y-axis starts at 0 for clarity
    ax.set_xlim(prices.min(), prices.max())
    ax.grid(True, alpha=0.3)
    _save_plot(fig, save_path)
    print(f'Saved {save_path}')


def _save_plot(fig, save_path):
    out_dir = os.path.dirname(save_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)