"""Generate the E2/E3 comparison and the E1-E4 overview.

All curves are unfiltered and unsmoothed, by epoch and measured time.
Output format is PNG only.
"""
import argparse
import json
from pathlib import Path
from evaluation.plots import save_curve_suite


def generate_comparisons(results_root, output_dir=None):
    root = Path(results_root)
    output = Path(output_dir) if output_dir else root / "figures"
    modes = ("E1", "E2", "E3", "E4")
    paths = {mode: root / "best_results" / mode / "evaluation_history.json" for mode in modes}
    missing = [mode for mode, path in paths.items() if not path.exists()]
    if missing:
        raise ValueError(f"Evaluation histories are required for: {', '.join(missing)}")
    histories = {mode: json.loads(path.read_text()) for mode, path in paths.items()}
    outputs = save_curve_suite(
        {mode: histories[mode] for mode in ("E2", "E3")}, output, prefix="E2_E3", include_combined=False,
    )
    outputs += save_curve_suite(histories, output, prefix="E1_E2_E3_E4", include_combined=False)
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
