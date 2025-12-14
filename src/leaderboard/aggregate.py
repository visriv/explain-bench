import os
import shutil

def append_to_global_leaderboard(local_tsv_path, global_tsv_path):
    """
    Appends rows from a per-dataset results.tsv into the global leaderboard.tsv.
    Ensures header consistency.
    """
    if not os.path.exists(local_tsv_path):
        raise FileNotFoundError(local_tsv_path)

    os.makedirs(os.path.dirname(global_tsv_path), exist_ok=True)

    with open(local_tsv_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if not lines:
        return

    header, rows = lines[0], lines[1:]

    if not os.path.exists(global_tsv_path):
        # first time → copy entire file
        with open(global_tsv_path, "w", encoding="utf-8") as g:
            g.write(header)
            g.writelines(rows)
    else:
        # append rows only
        with open(global_tsv_path, "a", encoding="utf-8") as g:
            g.writelines(rows)
