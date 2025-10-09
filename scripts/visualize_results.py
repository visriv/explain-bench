import argparse, pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    args = ap.parse_args()
    df = pd.read_csv(args.input)
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
