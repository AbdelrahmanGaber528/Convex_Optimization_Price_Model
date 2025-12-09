import matplotlib.pyplot as plt
import seaborn as sns
import os


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


def plot_comparison_all_three(prices, rev_convex, rev_nonconvex, rev_restored,
    optimal_price, optimal_revenue, save_path='reports/revenue_comparison.png'):
    fig, ax = plt.subplots(figsize=(12, 7))
    

    ax.plot(prices, rev_convex, label='Original Convex', linewidth=2, alpha=0.7)
    ax.plot(prices, rev_nonconvex, '--', label='Non-Convex', linewidth=2, alpha=0.7)
    ax.plot(prices, rev_restored, ':', label='Restored Convex', linewidth=5, alpha=0.7)


    if optimal_price is not None:
        ax.scatter([optimal_price], [optimal_revenue], color='red', s=100, label=f'Optimal ${optimal_price:.2f}', zorder=5)

    ax.set_xlabel('Price')
    ax.set_ylabel('Revenue')
    ax.set_title('Price vs Revenue — Convex vs Non-Convex vs Restored')
    ax.legend()
    _save_plot(fig, save_path)
    print(f'Saved {save_path}')


def plot_separate_revenues(prices, rev_convex, rev_nonconvex, rev_restored, optimal_price, optimal_revenue, save_dir='reports'):
    # Plot 1: Convex Revenue
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    ax1.plot(prices, rev_convex, label='Convex Revenue', linewidth=2)
    if optimal_price is not None:
        ax1.scatter([optimal_price], [optimal_revenue], color='red', s=100, label=f'Optimal Price: ${optimal_price:.2f}', zorder=5)
    ax1.set_xlabel('Price')
    ax1.set_ylabel('Revenue')
    ax1.set_title('Price vs. Convex Revenue')
    ax1.legend()
    _save_plot(fig1, os.path.join(save_dir, 'convex_revenue.png'))

    # Plot 2: Non-Convex Revenue
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    ax2.plot(prices, rev_nonconvex, '--', label='Non-Convex Revenue', linewidth=2)
    ax2.set_xlabel('Price')
    ax2.set_ylabel('Revenue')
    ax2.set_title('Price vs. Non-Convex Revenue')
    ax2.legend()
    _save_plot(fig2, os.path.join(save_dir, 'nonconvex_revenue.png'))

    # Plot 3: Restored Convex Revenue
    fig3, ax3 = plt.subplots(figsize=(12, 7))
    ax3.plot(prices, rev_restored, ':', label='Restored Convex Revenue', linewidth=2)
    ax3.set_xlabel('Price')
    ax3.set_ylabel('Revenue')
    ax3.set_title('Price vs. Restored Convex Revenue')
    ax3.legend()
    _save_plot(fig3, os.path.join(save_dir, 'restored_convex_revenue.png'))


def _save_plot(fig, save_path):
    out_dir = os.path.dirname(save_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

