import os
import numpy as np
from .optimizer import PricingModel
from .config import Params as ConfigParams
from .visualizer import (
    plot_original_convex_revenue,
    plot_non_convex_revenue,
    plot_restored_convex_revenue
)

def _generate_all_plots(model, results, prices, revenue_original_convex, revenue_non_convex, revenue_restored_convex):
    """Generates and saves all plots, returning their filenames."""
    
    reports_dir = "reports"
    plot_filenames = []
    
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)

    # Plot Original Convex
    plot_path = os.path.join(reports_dir, "revenue_original_convex.png")
    plot_original_convex_revenue(
        model=model, results=results,
        prices_array=prices, revenue_original_convex=revenue_original_convex,
        save_path=plot_path
    )
    plot_filenames.append(os.path.basename(plot_path))

    # Plot Non-Convex
    plot_path = os.path.join(reports_dir, "revenue_non_convex.png")
    plot_non_convex_revenue(
        model=model,
        prices_array=prices, revenue_non_convex=revenue_non_convex,
        save_path=plot_path
    )
    plot_filenames.append(os.path.basename(plot_path))

    # Plot Restored Convex
    plot_path = os.path.join(reports_dir, "revenue_restored_convex.png")
    plot_restored_convex_revenue(
        model=model, results=results,
        prices_array=prices, revenue_restored_convex=revenue_restored_convex,
        save_path=plot_path
    )
    plot_filenames.append(os.path.basename(plot_path))

    print(f"[SUCCESS] {len(plot_filenames)} plots generated and saved to '{reports_dir}/'")
    
    return plot_filenames


def run_optimization_workflow():
    
    # 1. Initialize Configuration
    params_instance = ConfigParams() 

    # 2. Initialize and Solve Model
    model = PricingModel()
    res_list = model.solve_convex()
    
    convexity_info = model.check_convexity()

    if convexity_info['curvature'] != 'CONCAVE':
        print("\n" + "="*80)
        print("WARNING: The model is not CONCAVE. Current convex maximization approach may not be suitable.")
        print("Proceeding with caution. Results might not represent a global maximum.")
        print("="*80 + "\n")

    # 3. Process Results
    results = {
        'price': res_list[0],
        'revenue': res_list[1],
        'demand': res_list[2],
        'status': res_list[3]
    }
    
    # 4. Prepare data for plotting all three cases
    max_price_for_plot = max(model.max_price * 1.2, results['price'] * 1.5) 
    prices = np.linspace(0, max_price_for_plot, 500)

    revenue_original_convex = model.calculate_convex_revenue(prices)
    revenue_non_convex = model.calculate_non_convex_revenue(prices)
    revenue_restored_convex = model.calculate_convex_revenue(prices) 

    # 5. Generate and Save Visualizations
    _generate_all_plots(model, results, prices, revenue_original_convex, revenue_non_convex, revenue_restored_convex)

    print("\n--- Optimization Results Summary ---")
    print(f"Optimal Price: ${results['price']:.2f}")
    print(f"Projected Revenue: ${results['revenue']:.2f}")
    print(f"Expected Demand: {results['demand']:.2f} units")
    print(f"Model Status: {results['status']}")
    print("Plots saved in the 'reports/' directory.")
    print("------------------------------------\n")


if __name__ == "__main__":
    run_optimization_workflow()