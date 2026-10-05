"""Reproducible bounded E3 -> E4 search and matched-budget E1/E2 comparison.

Run: python -m experiments.tune_cifar100 --output-dir results/tuning_cifar100_DATE
Overrides are process-local; config and experiment defaults are never edited.
Each trial uses the experiment runner's original result-saving logic.
"""
import argparse
import gc
import json
from pathlib import Path

import torch

from experiments import adaptive_lr as e3, adaptive_combined as e4
from experiments import fixed_fixed as e1, adaptive_batch as e2


def convergence(history):
    n = len(history)
    previous = min(r['val_loss'] for r in history[:-20])
    recent = min(r['val_loss'] for r in history[-20:])
    improvement = max(0., previous - recent)
    relative = improvement / previous
    changes = [r['epoch'] for r in history if r.get('lr_change_reason') in ('worsening_decay', 'plateau_decay')]
    return {'epochs': n, 'best_loss_before_last20': previous,
            'best_loss_last20': recent, 'absolute_improvement': improvement,
            'relative_improvement': relative, 'last_decay_epoch': changes[-1] if changes else None,
            'extend_budget': relative > .005 or (bool(changes) and changes[-1] > n-5 and improvement > .005)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    shared = dict(reference_batch_size=32, alpha=.2, lr_factor=.5,
                  plateau_patience=5, worsening_patience=3, cooldown_epochs=2,
                  warmup_epochs=0, min_lr=1e-6, min_delta_loss=.001, min_delta_acc=.002)

    def trial(label, mode, epochs, settings):
        trial_root = root/'trials'/label
        trial_root.mkdir(parents=True, exist_ok=True)
        print(f'START {label}: {epochs} epochs, {settings}', flush=True)
        if mode=='E3':
            tokens=['--dataset','cifar100','--epochs',str(epochs),'--seed','42',
                    '--batch-size','32','--output-dir',str(trial_root)]
            for k,v in settings.items():tokens += ['--'+k.replace('_','-'),str(v)]
            result = e3.run_adaptive_experiment(e3.parse_arguments(tokens))
        elif mode=='E4':
            e4.EPOCHS=epochs
            mapping={'reference_batch_size':'LR_REFERENCE_BATCH_SIZE','reference_lr':'LR_REFERENCE_LR',
                     'alpha':'LR_ALPHA','lr_factor':'LR_FACTOR','plateau_patience':'LR_PLATEAU_PATIENCE',
                     'worsening_patience':'LR_WORSENING_PATIENCE','cooldown_epochs':'LR_COOLDOWN_EPOCHS',
                     'warmup_epochs':'LR_WARMUP_EPOCHS','min_lr':'LR_MIN','min_delta_loss':'LR_MIN_DELTA_LOSS',
                     'min_delta_acc':'LR_MIN_DELTA_ACC'}
            for k,v in settings.items():setattr(e4,mapping[k],v)
            # Explicitly match the baseline reference in the module's frozen-setting guard.
            e4.INITIAL_LEARNING_RATE=settings['reference_lr']
            e4.INITIAL_BATCH_SIZE=32
            e4.ADAPTIVE_BATCH_OPTIONS=[32,64,128,256]
            e4.BATCH_CV_WINDOW=3;e4.BATCH_STABILITY_THRESHOLD=.25
            e4.BATCH_PLATEAU_PATIENCE=3;e4.BATCH_MIN_DELTA=.01;e4.BATCH_COOLDOWN_EPOCHS=2
            result=e4.run_experiment(e4.parse_arguments(['--dataset','cifar100','--epochs',str(epochs),
                            '--seed','42','--pilot','--output-dir',str(trial_root)]))
        else:
            module=e1 if mode=='E1' else e2
            module.RESULTS_DIR=trial_root
            module.INITIAL_LEARNING_RATE=settings['reference_lr']
            if mode=='E1':result=module.run_fixed_experiment(32,epochs,42,dataset='cifar100')
            else:
                module.EPOCHS=epochs;module.ADAPTIVE_BATCH_OPTIONS=[32,64,128,256]
                module.CV_WINDOW=3;module.STABILITY_THRESHOLD=.25
                module.PLATEAU_PATIENCE=3;module.MIN_DELTA=.01;module.COOLDOWN_EPOCHS=2
                result=module.run_experiment(epochs,42,pilot=True,dataset='cifar100')
        run_dir=Path(result['output_dir'])
        metrics=json.loads((run_dir/'validation_metrics.json').read_text())
        metadata=json.loads(next(run_dir.glob('*_metadata.json')).read_text())
        record={'label':label,'experiment':mode,'epochs':epochs,'settings':settings,
                'output_dir':str(run_dir),'validation_loss':metrics['loss'],
                'validation_accuracy':metrics['accuracy'],'validation_macro_f1':metrics['macro_f1'],
                'best_epoch':metrics['checkpoint_epoch'],
                'optimizer_updates':metadata['total_optimizer_updates'],
                'training_seconds':metadata['total_training_time_seconds'],
                'convergence':convergence(result['history'])}
        print(f"DONE {label}: loss={record['validation_loss']:.5f}, accuracy={100*record['validation_accuracy']:.2f}%, best epoch={record['best_epoch']}",flush=True)
        del result
        gc.collect();torch.cuda.empty_cache()
        return record

    candidates=[]
    for lr,patience in [(.005,3),(.01,3),(.02,3),(.01,5)]:
        settings={**shared,'reference_lr':lr,'worsening_patience':patience}
        candidates.append(trial(f'E3_lr{lr}_w{patience}_ep120','E3',120,settings))
    def rank(records):return min(records,key=lambda r:(r['validation_loss'],-r['validation_accuracy']))
    winner=rank(candidates)
    budget=120
    for next_budget in [160,200]:
        if not winner['convergence']['extend_budget']:break
        budget=next_budget
        winner=trial(f"E3_final_ep{budget}",'E3',budget,winner['settings'])
    # At the chosen budget, confirm the runner-up when the screening budget changed.
    if budget!=120:
        other=rank([c for c in candidates if c['settings']!=winner['settings']])
        finalist=trial(f'E3_runnerup_ep{budget}','E3',budget,other['settings'])
        winner=rank([winner,finalist])
    e4_candidates=[]
    for alpha in [0.,.2,.5]:
        settings={**winner['settings'],'alpha':alpha}
        e4_candidates.append(trial(f'E4_alpha{alpha}_ep{budget}','E4',budget,settings))
    best_e4=rank(e4_candidates)
    # E1/E2 share E3's initial batch, starting LR, model, optimizer and budget.
    baseline={'reference_lr':winner['settings']['reference_lr']}
    first=trial(f'E1_final_ep{budget}','E1',budget,baseline)
    second=trial(f'E2_final_ep{budget}','E2',budget,baseline)
    selected={'E1':first,'E2':second,'E3':winner,'E4':best_e4}
    for mode, record in selected.items():
        print(f"SELECTED {mode}: {record}", flush=True)
    print('SEARCH COMPLETED '+str(root),flush=True)


if __name__=='__main__':
    main()
