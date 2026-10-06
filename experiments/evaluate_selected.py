"""Evaluate already-selected E1-E4 checkpoints once on official CIFAR-100 test.

Selection is read-only: this command never changes settings, epoch budget or
checkpoint selection based on test metrics. Run after the validation search.
"""
import argparse
import csv
import json
from pathlib import Path
import torch
from torch.torch_version import TorchVersion
from config import DEVICE, set_seed
from data.cifar100 import CIFAR100DataModule
from models.custom_cnn import CustomCNN
from evaluation.metrics import evaluate_model
from evaluation.confusion_matrix import plot_confusion_matrix


def evaluate_selected(root):
    root=Path(root)
    selected=json.loads((root/'best_parameters.json').read_text())
    set_seed(selected['seed'])
    data=CIFAR100DataModule()
    loader=data.get_test_loader(batch_size=256)
    class_names=list(data.test_dataset.classes)
    results=[]
    for mode in ('E1','E2','E3','E4'):
        directory=root/'best_results'/mode
        checkpoint_path=next(directory.glob('*best.pt'))
        with torch.serialization.safe_globals([TorchVersion]):
            checkpoint=torch.load(checkpoint_path,map_location=DEVICE,weights_only=True)
        if checkpoint['epoch']!=selected['selected'][mode]['best_epoch']:
            raise ValueError('Checkpoint differs from frozen validation selection')
        model=CustomCNN(num_classes=len(class_names)).to(DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        metrics=evaluate_model(model,loader,DEVICE,class_names=class_names)
        payload={'split':'test','checkpoint_epoch':checkpoint['epoch'],
                 'class_names':class_names,'test_set_used_for_selection':False,**metrics}
        (directory/'test_metrics.json').write_text(json.dumps(payload,indent=2,allow_nan=False))
        plot_confusion_matrix(metrics['y_true'],metrics['y_pred'],
            directory/'test_confusion_matrix_normalized.png',normalize=True,
            class_names=class_names,title=f'{mode}: test confusion matrix (normalized)')
        results.append({'experiment':mode,'checkpoint_epoch':checkpoint['epoch'],
            'test_loss':metrics['loss'],'test_accuracy':metrics['accuracy'],'test_macro_f1':metrics['macro_f1']})
        print(f"{mode}: frozen checkpoint epoch {checkpoint['epoch']}; test accuracy {metrics['accuracy']:.2%}",flush=True)
        del checkpoint,model
        torch.cuda.empty_cache()
    with (root/'test_comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root',type=Path,required=True)
    args=parser.parse_args()
    evaluate_selected(args.results_root)


if __name__=='__main__':
    main()
