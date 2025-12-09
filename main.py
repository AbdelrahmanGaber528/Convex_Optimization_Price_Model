import argparse
from src.core.workflow import run_workflow

def main():
    parser = argparse.ArgumentParser(description='Run the pricing optimization workflow.')
    parser.add_argument('--dataset', type=str, help='Path to the dataset CSV file.')
    args = parser.parse_args()

    run_workflow(dataset_path='sales_data.csv')

if __name__ == '__main__':
    main()
