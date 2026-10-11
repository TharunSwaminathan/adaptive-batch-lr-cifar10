"""Regenerate E3/E4 and E1-E4 figures, separately by epoch and measured time.

python -m experiments.plot_comparison --results-root results/rerun_cifar100_20261006
All histories remain unfiltered and unsmoothed. Each loss/accuracy figure is
saved separately as PNG and vector PDF, alongside the paired comparison.
"""
import argparse
import json
from pathlib import Path
from evaluation.plots import save_curve_suite


def generate_comparisons(results_root, output_dir=None):
    root = Path(results_root)
    output = Path(output_dir) if output_dir else root / "figures"
    histories = {}
    for mode in ("E1", "E2", "E3", "E4"):
        path = root / "best_results" / mode / "evaluation_history.json"
        if path.exists():
            histories[mode] = json.loads(path.read_text())
            save_curve_suite({mode: histories[mode]}, path.parent)
    if not all(mode in histories for mode in ("E3", "E4")):
        raise ValueError("E3 and E4 evaluation histories are required")
    outputs = save_curve_suite({mode: histories[mode] for mode in ("E3", "E4")}, output, prefix="E3_E4")
    if len(histories) == 4:
        outputs += save_curve_suite(histories, output, prefix="E1_E4")
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    for path in generate_comparisons(args.results_root, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
