import importlib
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

run_training = importlib.import_module("convnext_pipeline.main").run_training


if __name__ == "__main__":
    run_training()