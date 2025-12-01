import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.workflow import run_optimization_workflow


if __name__ == "__main__":
    run_optimization_workflow()