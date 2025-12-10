import numpy as np
import pandas as pd
from .dataset_operations import Dataset
from .optimizer import PricingModel
from ..visuals.visualizer import plot_separate_revenues, plot_dataset_revenue, plot_dataset_convexity, plot_make_dataset_convex, plot_solver_objective
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
    print("Dataset Convexity Before Transformation:")
    print(f"  Is Convex: {convexity_info['is_convex']}")
    if not convexity_info['is_convex']:
        print(f"  Reason: Dataset is not convex and requires transformation.")

    # 2a. Make it convex if it is not
    if not convexity_info['is_convex']:
        print("\nStep 2a: Making the dataset convex...")
        plot_make_dataset_convex(prices, demands)
        dataset.make_convex()
        prices = dataset.prices
        demands = dataset.demands
        
        # Add a check after transformation
        print("\nStep 2b: Checking dataset convexity after transformation...")
        post_transform_convexity_info = dataset.check_dataset_convexity_convex_hull()
        print("Dataset Convexity After Transformation:")
        print(f"  Is Convex: {post_transform_convexity_info['is_convex']}")
        if post_transform_convexity_info['is_convex']:
            print("  Conclusion: Dataset successfully made convex.")
        else:
            print("  Warning: Dataset is still not convex after transformation.")

    # 3. Build model
    print("\nStep 3: Building optimization model...")
    model = PricingModel()
    model.max_price = Params.MAX_PRICE
    model._build_concave_model()

    prices_range = np.array(sorted(prices))
    concave_revenue = model.calculate_concave_revenue(prices_range)

    # 4. Check cvxpy & hessian
    print("\nStep 4: Checking convexity with CVXPY and Hessian...")
    cvxpy_check = model.check_convexity_curve_dcp()
    print("CVXPY Convexity Check:")
    print(f"  Curvature: {cvxpy_check['curvature']}")
    print(f"  DCP Compliant: {cvxpy_check['is_dcp_compliant']}")
    print(f"  Conclusion: {cvxpy_check['conclusion']}")
    hessian_check = model.check_convexity_hessian()
    print("Hessian Convexity Check:")
    print(f"  Result: {hessian_check['result']}")
    print(f"  Suitable for Maximization: {hessian_check['suitable_for_maximization']}")

    # 5. Solve concave problem
    print("\nStep 5: Solving concave optimization problem...")
    optimal_price, optimal_revenue, final_demand, status = model.solve_concave()
    print(f"Status: {status}")

    if optimal_price is not None:
        print(f"Optimal Price: {optimal_price:.2f}")
        print(f"Optimal Revenue: {optimal_revenue:.2f}")
        print(f"Final Demand: {final_demand:.2f}")
    else:
        print("Could not find optimal solution.")

    # 6. Turn it into a convex model
    print("\nStep 6: Turning into a convex model...")
    model.build_convex_model()
    
    # Add a check after building convex model
    print("\nStep 6a: Checking convexity of the new convex model...")
    cvxpy_check_convex = model.check_convexity_curve_dcp()
    print("CVXPY Convexity Check (Convex Model):")
    print(f"  Curvature: {cvxpy_check_convex['curvature']}")
    print(f"  DCP Compliant: {cvxpy_check_convex['is_dcp_compliant']}")
    print(f"  Conclusion: {cvxpy_check_convex['conclusion']}")
    hessian_check_convex = model.check_convexity_hessian()
    print("Hessian Convexity Check (Convex Model):")
    print(f"  Result: {hessian_check_convex['result']}")
    print(f"  Suitable for Maximization: {hessian_check_convex['suitable_for_maximization']}")

    # 7. Solve convex problem
    print("\nStep 7: Solving convex optimization problem...")
    model.solve_convex()

    # 8. Make it nonconvex
    print("\nStep 8: Making the model non-convex...")
    model.build_nonconvex_model()
    non_convex_revenue = model.calculate_nonconvex_revenue(prices_range)

    # 9. Restore convex
    print("\nStep 9: Restoring the convex model...")
    model.restore_convex_model()
    restored_revenue = model.calculate_concave_revenue(prices_range)

    # 10. Save plot to reports
    print("\nStep 10: Generating and saving comparison plot...")
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
    
    # 11. Plot solver objective
    print("\nStep 11: Generating and saving solver objective plot...")
    solver_objective = model.calculate_convex_revenue(prices_range)
    plot_solver_objective(prices_range, solver_objective)

    print("Workflow finished.")