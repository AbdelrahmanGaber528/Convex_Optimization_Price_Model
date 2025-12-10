import numpy as np
import pandas as pd
from .dataset_operations import Dataset
from .optimizer import PricingModel
from ..visuals.visualizer import plot_separate_revenues, plot_dataset_revenue, plot_dataset_convexity, plot_make_dataset_convex
from ..config import Params

def run_workflow(dataset_path=None):

    # 1. Load dataset
    print("\nStep 1: Loading dataset...")
    if dataset_path:
        df = pd.read_csv(dataset_path)
        prices = df['price'].values
        demands = df['demand'].values
    else:
        prices = np.array([5, 10, 15, 20, 25, 30, 35])
        demands = np.array([115, 105, 92, 70, 50, 30, 10])

    plot_dataset_revenue(prices, demands)

    # 2. Check if it's convex
    print("\nStep 2: Checking dataset convexity...")
    dataset = Dataset(prices, demands)
    convexity_info = dataset.check_dataset_convexity_convex_hull()
    plot_dataset_convexity(prices, demands)
    print(f"Curvature: {dataset.curvature}")

    # 2a. Make it convex if it is not
    if not convexity_info['is_convex']:
        print("\nStep 2a: Making the dataset convex...")
        plot_make_dataset_convex(prices, demands)
        dataset.make_convex()
        prices = dataset.prices
        demands = dataset.demands

    # 3. Build model
    print("\nStep 3: Building optimization model...")
    model = PricingModel()
    model.max_price = Params.MAX_PRICE
    model._build_concave_model()

    prices_range = np.array(sorted(prices))
    concave_revenue = model.calculate_concave_revenue(prices_range)

    # 4. Solve convex problem
    print("\nStep 4: Solving convex optimization problem...")
    optimal_price, optimal_revenue, final_demand, status = model.solve_concave()
    print(f"Status: {status}")

    if optimal_price is not None:
        print(f"Optimal Price: {optimal_price:.2f}")
        print(f"Optimal Revenue: {optimal_revenue:.2f}")
        print(f"Final Demand: {final_demand:.2f}")
    else:
        print("Could not find optimal solution.")

    # 5. Check cvxpy & hessian
    print("\nStep 5: Checking convexity with CVXPY and Hessian...")
    cvxpy_check = model.check_convexity_curve_dcp()
    print(f"CVXPY conclusion: {cvxpy_check['conclusion']}")
    hessian_check = model.check_convexity_hessian()
    print(f"Hessian result: {hessian_check['result']}")

    # 6. Make it nonconvex
    print("\nStep 6: Making the model non-convex...")
    model.build_nonconvex_model()
    non_convex_revenue = model.calculate_nonconvex_revenue(prices_range)

    # 7. Restore convex
    print("\nStep 7: Restoring the convex model...")
    model.restore_convex_model()
    restored_revenue = model.calculate_concave_revenue(prices_range)

    # 8. Save plot to reports
    print("\nStep 8: Generating and saving comparison plot...")
    if optimal_price is not None:
        plot_separate_revenues(
            prices_range,
            concave_revenue,
            non_convex_revenue,
            restored_revenue,
            optimal_price,
            optimal_revenue,
            save_dir='reports'
        )
    print("Workflow finished.")