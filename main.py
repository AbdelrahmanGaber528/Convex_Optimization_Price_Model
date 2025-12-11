import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from src.core.workflow import run_workflow

def main():
    run_workflow()

if __name__ == '__main__':
    main()
