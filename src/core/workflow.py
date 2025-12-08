import numpy as np
from src.load_dataset import load_sample_dataset, load_from_csv
from src.core.check_data_convex import DatasetConvexityChecker
from src.core.optimizer import PricingModel
from src.visuals.visualizer import plot_comparison_all_three

def run_workflow(dataset_path=None):

    # 1. Load dataset
    print("\nStep 1: Loading dataset...")
    if dataset_path:
        prices, demands = load_from_csv(dataset_path)
    else:
        prices, demands = load_sample_dataset()
    # 2. Check if it's convex
    print("\nStep 2: Checking dataset convexity...")
    checker = DatasetConvexityChecker(prices, demands)
    convexity_info = checker.check_curvature()
    print(f"Curvature: {convexity_info['curvature']}")

    # 3. Build model
    print("\nStep 3: Building optimization model...")
    model = PricingModel()
    model._build_model()

    prices_range = np.linspace(model.min_price, model.max_price, 20)
    convex_revenue = model.calculate_convex_revenue(prices_range)

    # 4. Solve convex problem
    print("\nStep 4: Solving convex optimization problem...")
    optimal_price, optimal_revenue, final_demand, status = model.solve_convex()
    print(f"Status: {status}")

    if optimal_price is not None:
        print(f"Optimal Price: {optimal_price:.2f}")
        print(f"Optimal Revenue: {optimal_revenue:.2f}")
        print(f"Final Demand: {final_demand:.2f}")
    else:
        print("Could not find optimal solution.")

    # 5. Check cvxpy & hessian
    print("\nStep 5: Checking convexity with CVXPY and Hessian...")
    cvxpy_check = model.check_convexity_cvxpy()
    print(f"CVXPY conclusion: {cvxpy_check['conclusion']}")
    hessian_check = model.check_convexity_hessian()
    print(f"Hessian result: {hessian_check['result']}")


    # 6. Make it nonconvex
    print("\nStep 6: Making the model non-convex...")
    model._build_nonconvex_model()
    non_convex_revenue = model.calculate_non_convex_revenue(prices_range)


    # 7. Restore convex
    print("\nStep 7: Restoring the convex model...")
    model.restore_convex_model()
    restored_revenue = model.calculate_convex_revenue(prices_range)


    # 8. Save comparison plot to reports
    print("\nStep 8: Generating and saving comparison plot...")
    if optimal_price is not None:
        plot_comparison_all_three(
            prices_range,
            convex_revenue,
            non_convex_revenue,
            restored_revenue,
            optimal_price,
            optimal_revenue,
            save_path='reports/revenue_comparison.png'
        )