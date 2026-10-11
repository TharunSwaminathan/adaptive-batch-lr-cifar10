# Adaptive CNN Visualizer

A Vite and Chart.js dashboard for replaying recorded CIFAR-10 and CIFAR-100 training runs and comparing fixed/adaptive learning-rate and batch-size strategies.

## Run locally

Requires Node.js and npm.

```sh
npm ci
npm run dev
```

Run the analysis tests with `npm test`; create a production build with `npm run build`.

## Experiment data

The included CSV snapshots are in `public/data/`. To download them again, run `python3 download_results.py` from this directory. The script fetches experiment outputs from the `jiong` branch of this repository.

Elapsed seconds are the recorded training times for these runs, not isolated hyperparameter-tuning durations. The comparisons are single-run validation observations and do not establish statistical significance or hardware-independent timing.
