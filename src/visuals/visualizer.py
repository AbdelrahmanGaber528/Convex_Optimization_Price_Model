import matplotlib.pyplot as plt
import seaborn as sns
import os


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




def _save_plot(fig, save_path):
    out_dir = os.path.dirname(save_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

