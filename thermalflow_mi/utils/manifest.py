"""Reconstruction manifests: one CSV row per private image of an attack run."""
import csv
from pathlib import Path

FIELDS = ["global_index", "class_index", "class_name", "condition_label", "source_path", "original_path",
          "observed_path", "reconstructed_path"]


def write_manifest(path, rows, extra_fields=()):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS + list(extra_fields))
        w.writeheader()
        w.writerows(rows)


def read_manifest(path):
    """Rows with image paths made absolute (paths are stored relative to the manifest's folder)."""
    path = Path(path)
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("original_path", "observed_path", "reconstructed_path"):
            if r.get(k) and not Path(r[k]).is_absolute():
                r[k] = str(path.parent / r[k])
    return rows
