#!/usr/bin/env python3
"""Regenerate a split CSV (the paper's splits are already in splits/; see thermalflow_mi/data/splits.py).

    python scripts/make_splits.py --dataset charlotte --out splits/charlotte_new.csv
"""
import argparse

import _bootstrap  # noqa: F401

from thermalflow_mi.config import load_dataset_config
from thermalflow_mi.data.splits import charlotte_split, cftd_split, thermvision_split, write_split


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["cftd", "charlotte", "thermvision"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = load_dataset_config(a.dataset).data_root
    rows = {"cftd": cftd_split, "charlotte": charlotte_split, "thermvision": thermvision_split}[a.dataset](root, a.seed)
    write_split(a.out, rows)
    print(f"{len(rows)} images -> {a.out}")


if __name__ == "__main__":
    main()
