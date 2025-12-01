import matplotlib.pyplot as plt
import seaborn as sns
import os
from .optimizer import PricingModel

def _save_plot(fig, save_path):
    output_dir = os.path.dirname(save_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(save_path)
    plt.close(fig)

def plot_original_convex_revenue(
    model: PricingModel, results: dict, prices_array, revenue_original_convex, save_path=None
):
    """Plots the Original (Convex) Revenue vs. Price."""
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))

    alpha = model.alpha
    beta = model.beta
    max_capacity = model.max_capacity
    optimal_price = results['price']
    optimal_revenue = results['revenue']

    ax.plot(prices_array, revenue_original_convex, label='Original Model (Convex)', 
             color='#1f77b4', linewidth=3)
    
    ax.scatter([optimal_price], [optimal_revenue], color='red', s=150, zorder=5, 
                label=f'Optimal: ${optimal_price:.2f}')
    
    limit_price = (alpha - max_capacity) / beta
    ax.axvline(x=limit_price, color='gray', linestyle=':', linewidth=2,
                label=f'Capacity Limit (Price > ${limit_price:.0f})')

    ax.set_title("Optimization Surface: Original (Convex)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Price ($)")
    ax.set_ylabel("Revenue ($)")
    ax.legend()
    plt.tight_layout()
    _save_plot(fig, save_path)


def plot_non_convex_revenue(
    model: PricingModel, prices_array, revenue_non_convex, save_path=None
):
    """Plots the Non-Convex Revenue vs. Price."""
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))

    alpha = model.alpha
    beta = model.beta
    max_capacity = model.max_capacity
    ax.plot(prices_array, revenue_non_convex, label='Non-Convex (Simulated Noise)', 
             color='#ff7f0e', linestyle='--', alpha=0.7)
    
    limit_price = (alpha - max_capacity) / beta
    ax.axvline(x=limit_price, color='gray', linestyle=':', linewidth=2,
                label=f'Capacity Limit (Price > ${limit_price:.0f})')

    ax.set_title("Optimization Surface: Non-Convex", fontsize=14, fontweight='bold')
    ax.set_xlabel("Price ($)")
    ax.set_ylabel("Revenue ($)")
    ax.legend()
    plt.tight_layout()
    _save_plot(fig, save_path)


def plot_restored_convex_revenue(
    model: PricingModel, results: dict, prices_array, revenue_restored_convex, save_path=None
):
    """Plots the Restored Convex Revenue vs. Price."""
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))

    alpha = model.alpha
    beta = model.beta
    max_capacity = model.max_capacity
    optimal_price = results['price']
    optimal_revenue = results['revenue']

    ax.plot(prices_array, revenue_restored_convex, label='Restored Convex', 
             color='green', linestyle=':', linewidth=2)
    
    ax.scatter([optimal_price], [optimal_revenue], color='red', s=150, zorder=5, 
                label=f'Optimal: ${optimal_price:.2f}')
    
    limit_price = (alpha - max_capacity) / beta
    ax.axvline(x=limit_price, color='gray', linestyle=':', linewidth=2,
                label=f'Capacity Limit (Price > ${limit_price:.0f})')

    ax.set_title("Optimization Surface: Restored Convex", fontsize=14, fontweight='bold')
    ax.set_xlabel("Price ($)")
    ax.set_ylabel("Revenue ($)")
    ax.legend()
    plt.tight_layout()
    _save_plot(fig, save_path)
