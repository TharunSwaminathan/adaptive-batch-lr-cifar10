"""Rerun saved settings with current default LR patience in a fresh directory."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parameters', type=Path, required=True)
    parser.add_argument('--experiment', choices=['E1', 'E2', 'E3', 'E4'], required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('results/reproduced_best'))
    args = parser.parse_args()
    saved = json.loads(args.parameters.read_text())
    mode = args.experiment
    record = saved['selected'][mode]
    epochs, seed = saved['epochs'], saved['seed']
    settings = dict(record['settings'])
    if mode in ('E3', 'E4'):
        from experiments.adaptive_lr import parse_arguments
        defaults = parse_arguments([])
        # Historical tuning must not override the user's current patience defaults.
        settings.update(plateau_patience=defaults.plateau_patience,
                        worsening_patience=defaults.worsening_patience)
    root = args.output_dir.resolve() / mode
    root.mkdir(parents=True, exist_ok=True)
    if mode == 'E3':
        from experiments import adaptive_lr as module
        tokens = ['--dataset', 'cifar100', '--epochs', str(epochs), '--seed', str(seed),
                  '--batch-size', '32', '--output-dir', str(root)]
        for key, value in settings.items():
            tokens += ['--' + key.replace('_', '-'), str(value)]
        module.run_adaptive_experiment(module.parse_arguments(tokens))
    elif mode == 'E4':
        from experiments import adaptive_combined as module
        mapping = {'reference_batch_size': 'LR_REFERENCE_BATCH_SIZE', 'reference_lr': 'LR_REFERENCE_LR',
                   'alpha': 'LR_ALPHA', 'lr_factor': 'LR_FACTOR', 'plateau_patience': 'LR_PLATEAU_PATIENCE',
                   'worsening_patience': 'LR_WORSENING_PATIENCE', 'cooldown_epochs': 'LR_COOLDOWN_EPOCHS',
                   'warmup_epochs': 'LR_WARMUP_EPOCHS', 'min_lr': 'LR_MIN',
                   'min_delta_loss': 'LR_MIN_DELTA_LOSS', 'min_delta_acc': 'LR_MIN_DELTA_ACC'}
        for key, value in settings.items():
            setattr(module, mapping[key], value)
        module.EPOCHS = epochs
        module.INITIAL_LEARNING_RATE = settings['reference_lr']
        module.INITIAL_BATCH_SIZE = 32
        module.ADAPTIVE_BATCH_OPTIONS = [32, 64, 128, 256]
        module.BATCH_CV_WINDOW = 3
        module.BATCH_STABILITY_THRESHOLD = .25
        module.BATCH_PLATEAU_PATIENCE = 3
        module.BATCH_MIN_DELTA = .01
        module.BATCH_COOLDOWN_EPOCHS = 2
        module.run_experiment(module.parse_arguments(['--dataset', 'cifar100', '--epochs', str(epochs),
                              '--seed', str(seed), '--pilot', '--output-dir', str(root)]))
    else:
        from experiments import fixed_fixed, adaptive_batch
        module = fixed_fixed if mode == 'E1' else adaptive_batch
        module.RESULTS_DIR = root
        module.INITIAL_LEARNING_RATE = settings['reference_lr']
        if mode == 'E1':
            module.run_fixed_experiment(32, epochs, seed, dataset='cifar100')
        else:
            module.EPOCHS = epochs
            module.ADAPTIVE_BATCH_OPTIONS = [32, 64, 128, 256]
            module.CV_WINDOW = 3
            module.STABILITY_THRESHOLD = .25
            module.PLATEAU_PATIENCE = 3
            module.MIN_DELTA = .01
            module.COOLDOWN_EPOCHS = 2
            module.run_experiment(epochs, seed, pilot=True, dataset='cifar100')


if __name__ == '__main__':
    main()
