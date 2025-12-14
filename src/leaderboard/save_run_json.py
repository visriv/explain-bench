import json, time, os

def save_run_json(run_root, flat_row):
    os.makedirs(run_root, exist_ok=True)
    fname = time.strftime("%Y-%m-%d_%H-%M-%S.jsonl")
    path = os.path.join(run_root, fname)
    with open(path, "a") as f:
        f.write(json.dumps(flat_row) + "\n")
