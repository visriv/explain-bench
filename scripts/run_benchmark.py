import argparse
from src.runner import BenchmarkRunner

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="explainbench/configs/default.yaml")
    args = ap.parse_args()
    runner = BenchmarkRunner(args.config)
    runner.run()

if __name__ == "__main__":
    main()
