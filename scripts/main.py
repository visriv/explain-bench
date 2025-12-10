import argparse
from src.runner import BenchmarkRunner
from src.datasets.datagen.spike_dataset import SpikeTrainDataset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    runner = BenchmarkRunner(args.config)
    runner.run()

if __name__ == "__main__":
    main()
