import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.core.dataset_operations import Dataset

def plot_dataset_revenue(prices, demands, save_path='reports/dataset_revenue.png'):
    """
    Plots the revenue from the raw dataset (price * demand).
    """
    revenue = prices * demands
    
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

def plot_separate_revenues(prices, rev_concave, rev_nonconvex, rev_restored, optimal_price, optimal_revenue, save_dir='reports'):
    """
    Plots separate graphs for concave, non-convex, and restored revenue models.
    """
    # Plot 1: Concave Revenue
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    ax1.plot(prices, rev_concave, label='Concave Revenue', linewidth=2)
    if optimal_price is not None:
        ax1.scatter([optimal_price], [optimal_revenue], color='red', s=100, label=f'Optimal Price: ${optimal_price:.2f}', zorder=5)
    ax1.set_xlabel('Price')
    ax1.set_ylabel('Revenue')
    ax1.set_title('Price vs. Concave Revenue')
    ax1.legend()
    _save_plot(fig1, os.path.join(save_dir, 'concave_revenue.png'))

    # Plot 2: Non-Convex Revenue
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    ax2.plot(prices, rev_nonconvex, '--', label='Non-Convex Revenue', linewidth=2)
    ax2.set_xlabel('Price')
    ax2.set_ylabel('Revenue')
    ax2.set_title('Price vs. Non-Convex Revenue')
    ax2.legend()
    _save_plot(fig2, os.path.join(save_dir, 'nonconvex_revenue.png'))

    # Plot 3: Restored Concave Revenue
    fig3, ax3 = plt.subplots(figsize=(12, 7))
    ax3.plot(prices, rev_restored, ':', label='Restored Concave Revenue', linewidth=2)
    ax3.set_xlabel('Price')
    ax3.set_ylabel('Revenue')
    ax3.set_title('Price vs. Restored Concave Revenue')
    ax3.legend()
    _save_plot(fig3, os.path.join(save_dir, 'restored_concave_revenue.png'))

def plot_dataset_convexity(prices, demands, save_path='reports/dataset_convexity.png'):
    """
    Checks and plots the convexity of the dataset.
    """
    dataset = Dataset(prices, demands)
    dataset.check_dataset_convexity_convex_hull()
    dataset.plot_hull(save_path=save_path)

def plot_make_dataset_convex(prices, demands, save_path='reports/made_convex_dataset.png'):
    """
    Makes the dataset convex and plots the result.
    """
    dataset = Dataset(prices, demands)
    dataset.make_convex()
    dataset.plot_hull(save_path=save_path)

def _save_plot(fig, save_path):
    out_dir = os.path.dirname(save_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)