"""Run E1-E4 in isolated processes with independent JSON settings."""
import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

MODES = ("E1", "E2", "E3", "E4")
COMMON = {"dataset": "cifar100", "model": "deeper_cnn_wide", "epochs": 60,
          "seed": 42, "batch_size": 32, "learning_rate": 0.01,
          "weight_decay": 0.001, "dropout_p": 0.2, "momentum": 0.9}
BATCH = {"batch_sizes": [32, 64, 128, 256], "cv_window": 3,
         "stability_threshold": 0.25, "plateau_patience": 3,
         "min_delta": 0.01, "cooldown_epochs": 2}
LR = {"reference_batch_size": 32, "alpha": 0.2, "lr_factor": 0.5,
      "plateau_patience": 5, "worsening_patience": 5, "cooldown_epochs": 3,
      "warmup_epochs": 0, "min_lr": 1e-5,
      "min_delta_loss": 0.001, "min_delta_acc": 0.002}


def load_settings(path):
    raw = json.loads(Path(path).read_text())
    if set(raw) - {"epochs"} != set(MODES):
        raise ValueError("Config must contain E1, E2, E3, E4 and optional shared epochs")
    shared_epochs = raw.get("epochs")
    if "epochs" in raw and (isinstance(shared_epochs, bool)
                            or not isinstance(shared_epochs, int) or shared_epochs < 1):
        raise ValueError("Shared epochs must be a positive integer")
    resolved = {}
    for mode in MODES:
        allowed = set(COMMON) | ({"batch_controller"} if mode in ("E2", "E4") else set()) | ({"lr_controller"} if mode in ("E3", "E4") else set())
        unknown = set(raw[mode]) - allowed
        if unknown:
            raise ValueError(f"{mode}: unknown settings {sorted(unknown)}")
        settings = {**COMMON, **raw[mode]}
        if shared_epochs is not None:
            settings["epochs"] = shared_epochs
        for key, defaults in (("batch_controller", BATCH), ("lr_controller", LR)):
            if key not in allowed:
                continue
            overrides = raw[mode].get(key, {})
            if set(overrides) - set(defaults):
                raise ValueError(f"{mode}: unknown {key} settings")
            settings[key] = {**defaults, **overrides}
        if settings['epochs'] < 1 or settings['batch_size'] < 1:
            raise ValueError(f"{mode}: epochs and batch_size must be positive")
        if settings['learning_rate'] <= 0 or settings['weight_decay'] < 0:
            raise ValueError(f"{mode}: invalid learning_rate or weight_decay")
        if not 0 <= settings['seed'] < 2**32 or not 0 <= settings['dropout_p'] < 1:
            raise ValueError(f"{mode}: invalid seed or dropout_p")
        if 'batch_controller' in settings and settings['batch_size'] not in settings['batch_controller']['batch_sizes']:
            raise ValueError(f"{mode}: initial batch_size must occur in batch_sizes")
        resolved[mode] = settings
    return resolved


def run_worker(mode, settings, root, *, validate_only=False):
    # Configure defaults before importing runners; child processes prevent
    # independently configured experiments from changing each other's state.
    import config
    config.INITIAL_BATCH_SIZE = settings['batch_size']
    config.INITIAL_LEARNING_RATE = settings['learning_rate']
    config.WEIGHT_DECAY = settings['weight_decay']
    config.MOMENTUM = settings['momentum']
    config.EPOCHS = settings['epochs']
    config.RESULTS_DIR = root / 'runs' / mode
    from functools import partial
    from models.factory import MODELS, create_model
    if settings['model'] != 'custom_cnn':
        MODELS[settings['model']] = partial(MODELS[settings['model']], dropout_p=settings['dropout_p'])
    model = create_model(settings['model'], 100 if settings['dataset'] == 'cifar100' else 10)
    from experiments.datasets import dataset_settings
    dataset_settings(settings['dataset'])
    import torch
    optimizer = torch.optim.SGD(model.parameters(), lr=settings['learning_rate'],
                                momentum=settings['momentum'], weight_decay=settings['weight_decay'])
    if 'batch_controller' in settings:
        from training.batch_controller import AdaptiveBatchController
        AdaptiveBatchController(initial_batch_size=settings['batch_size'], **settings['batch_controller'])
    if 'lr_controller' in settings:
        from training.lr_controller import LRController
        LRController(optimizer, mode='adaptive', reference_lr=settings['learning_rate'], **settings['lr_controller'])
    del model, optimizer
    if validate_only:
        return
    common = dict(epochs=settings['epochs'], seed=settings['seed'], dataset=settings['dataset'])
    if mode == 'E1':
        from experiments.fixed_fixed import run_fixed_experiment
        result = run_fixed_experiment(batch_size=settings['batch_size'], model_name=settings['model'], **common)
    elif mode == 'E2':
        from experiments import adaptive_batch as runner
        for key, value in settings['batch_controller'].items():
            target = 'ADAPTIVE_BATCH_OPTIONS' if key == 'batch_sizes' else key.upper()
            setattr(runner, target, value)
        result = runner.run_experiment(model_name=settings['model'], pilot=True, enforce_frozen=False, **common)
    elif mode == 'E3':
        from experiments.adaptive_lr import run_adaptive_experiment
        args = SimpleNamespace(**common, model=settings['model'], batch_size=settings['batch_size'],
                               reference_lr=settings['learning_rate'], output_dir=config.RESULTS_DIR,
                               target_accuracy=0.8, **settings['lr_controller'])
        result = run_adaptive_experiment(args)
    else:
        from experiments import adaptive_combined as runner
        for key, value in settings['batch_controller'].items():
            target = 'ADAPTIVE_BATCH_OPTIONS' if key == 'batch_sizes' else 'BATCH_' + key.upper()
            setattr(runner, target, value)
        mapping = {'reference_batch_size': 'LR_REFERENCE_BATCH_SIZE', 'alpha': 'LR_ALPHA',
                   'lr_factor': 'LR_FACTOR', 'min_lr': 'LR_MIN'}
        for key, value in settings['lr_controller'].items():
            setattr(runner, mapping.get(key, 'LR_' + key.upper()), value)
        runner.LR_REFERENCE_LR = settings['learning_rate']
        args = SimpleNamespace(**common, model=settings['model'], output_dir=config.RESULTS_DIR, pilot=True)
        result = runner.run_experiment(args, enforce_frozen=False)
    result_path = Path(result['output_dir']).resolve()
    link = root / 'best_results' / mode
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(result_path, target_is_directory=True)
    (root / f'{mode}_result.json').write_text(json.dumps({'output_dir': str(result_path)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('suite_config.json'))
    parser.add_argument('--output-dir', type=Path, default=Path('results/unified_experiments'))
    parser.add_argument('--dry-run', action='store_true', help='Validate all settings without loading datasets or training.')
    parser.add_argument('--worker', choices=MODES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    settings = load_settings(args.config)
    if args.worker:
        run_worker(args.worker, settings[args.worker], args.output_dir.resolve(), validate_only=args.dry_run)
        return
    root = args.output_dir.resolve()
    if not args.dry_run:
        root = root / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        root.mkdir(parents=True, exist_ok=False)
        (root / 'settings.json').write_text(json.dumps(settings, indent=2))
    for mode in MODES:
        command = [sys.executable, '-m', 'experiments.run_suite', '--config',
                   str(args.config.resolve()), '--output-dir', str(root), '--worker', mode]
        if args.dry_run:
            command.append('--dry-run')
            subprocess.run(command, check=True)
        else:
            print(f'Running {mode}; log: {root / (mode + ".log")}', flush=True)
            with (root / f'{mode}.log').open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True,
                               env=os.environ | {'PYTHONUNBUFFERED': '1'})
        print(f'{mode}: {"settings validated" if args.dry_run else "completed"}', flush=True)
    if args.dry_run:
        return
    from experiments.plot_comparison import generate_comparisons
    generate_comparisons(root)
    rows = []
    for mode in MODES:
        directory = root / 'best_results' / mode
        metric = json.loads((directory / 'validation_metrics.json').read_text())
        metadata = json.loads(next(directory.glob('*metadata.json')).read_text())
        rows.append(dict(experiment=mode, best_epoch=metric['checkpoint_epoch'],
                         validation_accuracy=metric['accuracy'], validation_loss=metric['loss'],
                         macro_f1=metric['macro_f1'], training_seconds=metadata['total_training_time_seconds']))
    with (root / 'comparison.csv').open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f'All experiments and comparisons saved to: {root}')


if __name__ == '__main__':
    main()
