"""Remove generated caches and, optionally, experiment outputs.

Dry-run by default. Use --delete to remove caches. Add --outputs to also clear
results/checkpoints while preserving their README files.
"""

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def targets(include_outputs=False):
    found = list(ROOT.rglob("__pycache__"))
    found += list(ROOT.rglob("*.pyc"))
    found += list(ROOT.rglob(".ipynb_checkpoints"))
    if include_outputs:
        for parent in (ROOT / "results", ROOT / "checkpoints"):
            if parent.exists():
                found += [p for p in parent.iterdir() if p.name != "README.md"]
    return sorted(set(found), key=lambda p: len(p.parts), reverse=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--delete", action="store_true")
    parser.add_argument("--outputs", action="store_true")
    args = parser.parse_args()

    items = targets(args.outputs)
    if not items:
        print("Nothing to clean.")
        return

    for path in items:
        print(("DELETE " if args.delete else "WOULD DELETE ") + str(path.relative_to(ROOT)))
        if args.delete and path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()


if __name__ == "__main__":
    main()
