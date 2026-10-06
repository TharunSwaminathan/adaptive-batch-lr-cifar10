"""Validation-only search with fixed default LR patience and matched final budget.

python -m experiments.rerun_cifar100 --output-dir results/rerun_cifar100_20261006
Screens LR/decay/batch scaling, checks 80/120/160/200-epoch budgets, then keeps
only one final run per E1-E4. Candidate scores and selection rationale survive
in manifest.json; trial model/history files are removed only after verification.
"""
import argparse
import contextlib
import csv
import gc
import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
import torch
from experiments import adaptive_lr as e3, adaptive_combined as e4
from experiments import adaptive_batch as e2, fixed_fixed as e1
from experiments.plot_comparison import generate_comparisons


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def plateau(history):
    """Require 20 epochs without >0.5% improvement in running-best val loss."""
    previous = min(r['val_loss'] for r in history[:-20])
    recent = min(r['val_loss'] for r in history[-20:])
    gain = max(0.0, (previous - recent) / previous)
    # Do not call a run settled if an impactful decay just happened.
    decays = [r['epoch'] for r in history if r.get('lr_change_reason') in ('plateau_decay', 'worsening_decay')]
    last_decay = decays[-1] if decays else None
    return {'last20_relative_loss_improvement': gain, 'last_decay_epoch': last_decay,
            'settled': gain <= .005 and not (last_decay and last_decay > len(history)-5 and gain > .002)}


def write_table(path, rows):
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--resume', action='store_true', help='Reuse verified completed trials from an interrupted search')
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=args.resume)
    defaults = e3.parse_arguments([])
    assert (defaults.plateau_patience, defaults.worsening_patience) == (2, 2)
    # B_ref=32 expresses LR directly as the actual initial LR, and makes
    # E3 invariant to alpha. The default B_ref=256 policy is an exact reparam.
    default_initial = defaults.reference_lr * (32/defaults.reference_batch_size)**defaults.alpha
    shared = dict(reference_batch_size=32, alpha=defaults.alpha, lr_factor=defaults.lr_factor,
                  plateau_patience=defaults.plateau_patience, worsening_patience=defaults.worsening_patience,
                  cooldown_epochs=defaults.cooldown_epochs, warmup_epochs=defaults.warmup_epochs,
                  min_lr=defaults.min_lr, min_delta_loss=defaults.min_delta_loss, min_delta_acc=defaults.min_delta_acc)
    records = []
    status_path = root/'status.json'
    started = datetime.now(timezone.utc).isoformat()
    if args.resume and (root/'search_plan.json').exists():
        started = json.loads((root/'search_plan.json').read_text())['started_utc']
    save_json(root/'search_plan.json', {'dataset':'CIFAR-100','seed':42,'started_utc':started,
        'patience_fixed':[2,2], 'initial_lr_candidates':[default_initial,.01,.02],
        'decay_factor_candidates':[.5,.7], 'joint_initial_lr_decay_check':True, 'E4_alpha_candidates':[0,.2,.5,1.],
        'epoch_budgets':[80,120,160,200], 'selection':'minimum validation loss, accuracy breaks ties',
        'epoch_rule':'first common budget with <=0.5% running-best val-loss gain in last20 for E3 and E4',
        'reference_batch_reparameterization':'B_ref=32; first candidate matches current default initial LR',
        'test_set_used':False})

    def trial(label, mode, epochs, settings):
        if args.resume:
            for meta_path in (root/'trials').rglob('*_metadata.json'):
                cached_meta=json.loads(meta_path.read_text())
                cached_dir=meta_path.parent
                if cached_meta.get('epochs')!=epochs or not (cached_dir/'validation_metrics.json').exists():
                    continue
                expected_type={'E3':'fixed_batch_adaptive_lr','E4':'adaptive_batch_adaptive_lr'}
                if mode not in expected_type or cached_meta.get('experiment_type')!=expected_type[mode]:
                    continue
                if cached_meta.get('lr_controller')!=settings:
                    continue
                cached_rows=list(csv.DictReader(next(cached_dir.glob('*.csv')).open()))
                if len(cached_rows)!=epochs or not (cached_dir/'lr_history.json').exists():
                    continue
                cached_history=[{**r,'epoch':int(r['epoch']),'val_loss':float(r['val_loss'])} for r in cached_rows]
                cached_metrics=json.loads((cached_dir/'validation_metrics.json').read_text())
                record={'label':label,'experiment':mode,'epochs':epochs,'settings':dict(settings),
                    'output_dir':str(cached_dir),'validation_loss':cached_metrics['loss'],
                    'validation_accuracy':cached_metrics['accuracy'],'validation_macro_f1':cached_metrics['macro_f1'],
                    'best_epoch':cached_metrics['checkpoint_epoch'],
                    'optimizer_updates':cached_meta['total_optimizer_updates'],
                    'training_seconds':cached_meta['total_training_time_seconds'],'convergence':plateau(cached_history)}
                records.append(record)
                print(f"REUSE {label}: loss={record['validation_loss']:.5f}, source={cached_dir}",flush=True)
                return record
        trial_root=root/'trials'/label
        if trial_root.exists():
            suffix=hashlib.sha256(json.dumps(settings,sort_keys=True).encode()).hexdigest()[:8]
            label=f'{label}_{suffix}'
            trial_root=root/'trials'/label
            if trial_root.exists():
                # A partial trial contains no completed result and can be replaced.
                shutil.rmtree(trial_root)
        save_json(status_path, {'status':'running','phase':'training','trial':label,'epochs':epochs})
        print(f'START {label}: {epochs} epochs, {settings}', flush=True)
        trial_root.mkdir(parents=True)
        with (trial_root/'console.log').open('w', buffering=1) as log, contextlib.redirect_stdout(log):
            if mode=='E3':
                tokens=['--dataset','cifar100','--epochs',str(epochs),'--seed','42',
                        '--batch-size','32','--output-dir',str(trial_root)]
                for k,v in settings.items(): tokens += ['--'+k.replace('_','-'),str(v)]
                result=e3.run_adaptive_experiment(e3.parse_arguments(tokens))
            elif mode=='E4':
                mapping={'reference_batch_size':'LR_REFERENCE_BATCH_SIZE','reference_lr':'LR_REFERENCE_LR',
                         'alpha':'LR_ALPHA','lr_factor':'LR_FACTOR','plateau_patience':'LR_PLATEAU_PATIENCE',
                         'worsening_patience':'LR_WORSENING_PATIENCE','cooldown_epochs':'LR_COOLDOWN_EPOCHS',
                         'warmup_epochs':'LR_WARMUP_EPOCHS','min_lr':'LR_MIN','min_delta_loss':'LR_MIN_DELTA_LOSS',
                         'min_delta_acc':'LR_MIN_DELTA_ACC'}
                overrides={mapping[k]:v for k,v in settings.items()}
                overrides.update(INITIAL_LEARNING_RATE=settings['reference_lr'])
                old={k:getattr(e4,k) for k in overrides}
                try:
                    for k,v in overrides.items(): setattr(e4,k,v)
                    result=e4.run_experiment(e4.parse_arguments(['--dataset','cifar100','--epochs',str(epochs),
                           '--seed','42','--pilot','--output-dir',str(trial_root)]))
                finally:
                    for k,v in old.items(): setattr(e4,k,v)
            elif mode=='E1':
                old=(e1.RESULTS_DIR,e1.INITIAL_LEARNING_RATE)
                try:
                    e1.RESULTS_DIR=trial_root; e1.INITIAL_LEARNING_RATE=settings['reference_lr']
                    result=e1.run_fixed_experiment(32,epochs,42,dataset='cifar100')
                finally: e1.RESULTS_DIR,e1.INITIAL_LEARNING_RATE=old
            else:
                old=(e2.RESULTS_DIR,e2.INITIAL_LEARNING_RATE)
                try:
                    e2.RESULTS_DIR=trial_root; e2.INITIAL_LEARNING_RATE=settings['reference_lr']
                    result=e2.run_experiment(epochs,42,pilot=True,dataset='cifar100')
                finally: e2.RESULTS_DIR,e2.INITIAL_LEARNING_RATE=old
        run_dir=Path(result['output_dir'])
        metrics=json.loads((run_dir/'validation_metrics.json').read_text())
        metadata=json.loads(next(run_dir.glob('*_metadata.json')).read_text())
        history=result['history']
        record={'label':label,'experiment':mode,'epochs':epochs,'settings':dict(settings),
                'output_dir':str(run_dir),'validation_loss':metrics['loss'],
                'validation_accuracy':metrics['accuracy'],'validation_macro_f1':metrics['macro_f1'],
                'best_epoch':metrics['checkpoint_epoch'],'optimizer_updates':metadata['total_optimizer_updates'],
                'training_seconds':metadata['total_training_time_seconds'],'convergence':plateau(history)}
        records.append(record)
        save_json(root/'manifest.json', {'status':'running','trials':records})
        print(f"DONE {label}: loss={record['validation_loss']:.5f}, acc={100*record['validation_accuracy']:.2f}%, best={record['best_epoch']}, settled={record['convergence']['settled']}", flush=True)
        del result
        gc.collect(); torch.cuda.empty_cache()
        return record

    def rank(candidates):
        return min(candidates,key=lambda r:(r['validation_loss'],-r['validation_accuracy']))

    candidates=[]
    for lr in [default_initial,.01,.02]:
        candidates.append(trial(f'E3_lr{lr:.6g}_factor0.5_ep80','E3',80,{**shared,'reference_lr':lr}))
    initial_winner=rank(candidates)
    candidates.append(trial('E3_slower_decay_factor0.7_ep80','E3',80,{**initial_winner['settings'],'lr_factor':.7}))
    # Changing decay can change the best initial LR: check the other rates too.
    for lr in [default_initial,.01,.02]:
        if lr==initial_winner['settings']['reference_lr']:
            continue
        candidates.append(trial(f'E3_lr{lr:.6g}_factor0.7_ep80','E3',80,
                                {**shared,'reference_lr':lr,'lr_factor':.7}))
    winner=rank(candidates)
    # Screen the runner-up for later improvement if its late curve is still moving.
    other=rank([r for r in candidates if r['label']!=winner['label']])
    if not winner['convergence']['settled'] or not other['convergence']['settled']:
        extended=[trial('E3_frontier1_ep120','E3',120,winner['settings']),
                  trial('E3_frontier2_ep120','E3',120,other['settings'])]
        winner=rank(extended)
    budget=winner['epochs']
    e4_candidates=[]
    for alpha in [0.,.2,.5,1.]:
        e4_candidates.append(trial(f'E4_alpha{alpha}_ep{budget}','E4',budget,{**winner['settings'],'alpha':alpha}))
    combined=rank(e4_candidates)
    # Both final adaptive runs share the exact policy. Alpha has no effect on
    # fixed-batch E3 because B=B_ref, but we rerun to verify matching metadata.
    final_policy=dict(combined['settings'])
    frontier=[]
    while budget < 200 and (not winner['convergence']['settled'] or not combined['convergence']['settled']):
        budget += 40
        winner=trial(f'E3_budget_ep{budget}','E3',budget,final_policy)
        combined=trial(f'E4_budget_ep{budget}','E4',budget,final_policy)
        frontier.append({'epochs':budget,'E3':winner['convergence'],'E4':combined['convergence']})
    selected={}
    if winner['settings']==final_policy and winner['epochs']==budget:
        selected['E3']=winner
    else:
        selected['E3']=trial(f'E3_final_ep{budget}','E3',budget,final_policy)
    selected['E4']=combined
    baseline={'reference_lr':final_policy['reference_lr']}
    selected['E1']=trial(f'E1_final_ep{budget}','E1',budget,baseline)
    selected['E2']=trial(f'E2_final_ep{budget}','E2',budget,baseline)
    save_json(status_path, {'status':'running','phase':'export','selected_epochs':budget})
    for mode,record in selected.items():
        destination=root/'best_results'/mode
        source_dir=Path(record['output_dir'])
        shutil.copytree(source_dir,destination)
        source_log=source_dir.parents[1]/'console.log'
        if source_log.exists():
            shutil.copy2(source_log,destination/'console.log')
        metadata_path=next(destination.glob('*_metadata.json'))
        retained_metadata=json.loads(metadata_path.read_text())
        retained_metadata['source_output_dir']=str(source_dir)
        retained_metadata['output_dir']=str(destination)
        save_json(metadata_path,retained_metadata)
        record['source_trial']=record['label']
        record['output_dir']=str(destination)
        save_json(destination/'selected_parameters.json', record)
    save_json(root/'best_parameters.json', {'epochs':budget,'seed':42,'dataset':'cifar100',
        'selected':selected,'shared_lr_policy':final_policy,'patience_source':'experiments.adaptive_lr defaults',
        'epoch_selection':{'frontier':frontier,'E3':selected['E3']['convergence'],'E4':selected['E4']['convergence']}})
    generate_comparisons(root)
    table=[]; thresholds=[]
    for mode in ('E1','E2','E3','E4'):
        r=selected[mode]
        table.append({'experiment':mode,'epochs':budget,'best_epoch':r['best_epoch'],
          'val_loss':r['validation_loss'],'val_accuracy':r['validation_accuracy'],
          'macro_f1':r['validation_macro_f1'],'optimizer_updates':r['optimizer_updates'],
          'training_seconds':r['training_seconds']})
        h=json.loads((root/'best_results'/mode/'evaluation_history.json').read_text())
        for threshold in [.30,.35,.40,.42,.44]:
            first=next((x for x in h if x['val_acc']>=threshold),None)
            sustained=next((h[i] for i in range(len(h)-4) if all(x['val_acc']>=threshold for x in h[i:i+5])),None)
            thresholds.append({'experiment':mode,'target_accuracy':threshold,
                'first_epoch':first['epoch'] if first else None,
                'first_seconds':first['elapsed_seconds'] if first else None,
                'sustained5_start_epoch':sustained['epoch'] if sustained else None,
                'sustained5_confirmed_seconds':h[h.index(sustained)+4]['elapsed_seconds'] if sustained else None})
    write_table(root/'comparison.csv',table)
    write_table(root/'time_to_accuracy.csv',thresholds)
    # Verify retained results before deleting candidate artifacts.
    checks={}
    for mode,record in selected.items():
        p=Path(record['output_dir']);meta=json.loads(next(p.glob('*metadata.json')).read_text())
        rows=list(csv.DictReader(next(p.glob('*.csv')).open()))
        metric=json.loads((p/'validation_metrics.json').read_text())
        assert len(rows)==budget
        best=min(rows,key=lambda x:float(x['val_loss']))
        assert int(best['epoch'])==metric['checkpoint_epoch']==record['best_epoch']
        assert math.isclose(float(best['val_loss']),metric['loss'],abs_tol=2e-6)
        if mode in ('E3','E4'):
            assert meta['lr_controller']==final_policy
            assert meta['lr_controller']['plateau_patience']==2
            assert meta['lr_controller']['worsening_patience']==2
        for axis in ('epoch','time'):
            for metric_name in ('loss','accuracy'):
                for extension in ('png','pdf'):
                    figure=p/f'{metric_name}_{axis}.{extension}'
                    assert figure.exists() and figure.stat().st_size>0
        checkpoint=next(p.glob('*best.pt'))
        checks[mode]={'epochs':len(rows),'best_epoch':record['best_epoch'],
                     'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                     'loss_matches_checkpoint_epoch':True}
    snapshot=root/'source_snapshot'
    project=Path(__file__).resolve().parents[1]
    for folder in ('experiments','training','evaluation','models','data'):
        for source in (project/folder).glob('*.py'):
            target=snapshot/folder/source.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    shutil.copy2(project/'config.py',snapshot/'config.py')
    save_json(root/'verification.json',{'passed':True,'checks':checks})
    # Keep compact screening evidence without paths pointing to deleted trials.
    evidence=[{k:v for k,v in r.items() if k!='output_dir'} for r in records]
    save_json(root/'manifest.json',{'status':'completed','started_utc':started,
        'finished_utc':datetime.now(timezone.utc).isoformat(),'screening_evidence':evidence,
        'selected':selected,'trials_removed':True,'test_set_used':False})
    report=['# CIFAR-100 rerun: default LR patience 2/2','',
      f'Common final budget: {budget} epochs; seed 42; validation-only selection; unchanged CNN/data split.',
      'Best means lowest validation loss among the tested candidates. Historical result files were not rewritten.',
      '', '## Final results','', '|Experiment|Epochs|Best checkpoint|Val loss|Val accuracy|Macro F1|Updates|Seconds|',
      '|---|---:|---:|---:|---:|---:|---:|---:|']
    for t in table:
        report.append(f"|{t['experiment']}|{budget}|{t['best_epoch']}|{t['val_loss']:.5f}|{100*t['val_accuracy']:.2f}%|{t['macro_f1']:.4f}|{t['optimizer_updates']}|{t['training_seconds']:.2f}|")
    report += ['', '## Shared E3/E4 policy','', '```json',json.dumps(final_policy,indent=2),'```',
        '', 'B_ref=32 makes reference_lr the actual starting LR. The default-equivalent initial-LR candidate was included.',
        'E3 and E4 have identical LR policy settings; only E4 changes batch. E2/E4 batch policy is unchanged (patience=3).',
        '', '## Epoch selection','',json.dumps({'E3':selected['E3']['convergence'],'E4':selected['E4']['convergence']},indent=2),
        'Budget grows by 40 until both adaptive runs have <=0.5% running-best val-loss gain over the last 20 epochs, up to 200.',
        'If a run remains unsettled at 200, this is a budget limit, not proof of convergence.',
        '', '## Figures and reproducibility','',
        'figures/E3_E4_training_curves.png and E3_E4_training_curves_time.png are paired comparisons.',
        'All loss and accuracy plots also have separate epoch/time PNG and PDF files, per experiment and per comparison.',
        'Time is recorded cumulative training+validation wall-clock time. Curves are raw and stop at each measured end.',
        'time_to_accuracy.csv contains first crossing and five-consecutive-epoch confirmation; never reached is blank.',
        'Only one selected run per experiment is retained. manifest.json keeps compact screening scores.',
        '', '```sh',
        'LD_LIBRARY_PATH=/home/ams098z/miniforge3/envs/py312/lib /home/ams098z/miniforge3/envs/py312/bin/python -m experiments.plot_comparison --results-root '+str(root),
        'LD_LIBRARY_PATH=/home/ams098z/miniforge3/envs/py312/lib /home/ams098z/miniforge3/envs/py312/bin/python -m experiments.reproduce_tuned --parameters '+str(root/'best_parameters.json')+' --experiment E3',
        '```','']
    (root/'REPORT.md').write_text('\n'.join(report))
    shutil.rmtree(root/'trials')
    save_json(status_path,{'status':'completed','phase':'complete','selected_epochs':budget,
                          'retained_experiments':['E1','E2','E3','E4'],'verification_passed':True})
    print('COMPLETED '+str(root),flush=True)


if __name__=='__main__':
    main()
