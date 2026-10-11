# Adaptive CNN Visualizer

A Vite and Chart.js dashboard for replaying and comparing the recorded CIFAR-10 and CIFAR-100 training runs from this repository.

## Run locally

Requires Node.js and npm.

```sh
npm ci
npm run dev
```

Create a production build with `npm run build`.

## Experiment data

The CSV snapshots used by the dashboard are included under `public/data/`. To download them again, run `python3 download_results.py` from this directory. The script reads the experiment outputs from the `jiong` branch of this repository.

Elapsed seconds represent recorded training time; they are not a direct measurement of hyperparameter-tuning overhead.
