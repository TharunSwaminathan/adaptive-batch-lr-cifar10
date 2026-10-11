"""Write a readable report from frozen selected results and measured histories."""
import argparse
import csv
import json
from pathlib import Path


def build_report(root):
    root=Path(root).resolve()
    params=json.loads((root/'best_parameters.json').read_text())
    manifest=json.loads((root/'manifest.json').read_text())
    selected=params['selected']
    policy=params['shared_lr_policy']
    epochs=params['epochs']
    tests={mode:json.loads((root/'best_results'/mode/'test_metrics.json').read_text())
           for mode in ('E1','E2','E3','E4') if (root/'best_results'/mode/'test_metrics.json').exists()}
    timing=list(csv.DictReader((root/'time_to_accuracy.csv').open()))
    def milestone(mode,target):
        return next(r for r in timing if r['experiment']==mode and float(r['target_accuracy'])==target)
    e3,e4=selected['E3'],selected['E4']
    reduction=100*(1-e4['training_seconds']/e3['training_seconds'])
    update_reduction=100*(1-e4['optimizer_updates']/e3['optimizer_updates'])
    lines=['# CIFAR-100 新参数实验结果','',
      f'推荐共同展示预算：**{epochs} epochs**。seed={params["seed"]}；训练45,000、验证5,000、官方测试10,000张；模型、数据划分、SGD和归一化保持一致。',
      '模型按最低验证 loss 选择 checkpoint；测试集仅在参数和预算冻结后评估，不参与重新选参。',
      '', '## 最终结果','',
      '|实验|预算|最佳checkpoint|验证loss|验证准确率|测试准确率|测试macro F1|训练秒数|更新次数|',
      '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for mode in ('E1','E2','E3','E4'):
        r=selected[mode];t=tests.get(mode)
        test_acc=f"{100*t['accuracy']:.2f}%" if t else '未评估'
        test_f1=f"{t['macro_f1']:.4f}" if t else '—'
        lines.append(f"|{mode}|{epochs}|{r['best_epoch']}|{r['validation_loss']:.5f}|{100*r['validation_accuracy']:.2f}%|{test_acc}|{test_f1}|{r['training_seconds']:.2f}|{r['optimizer_updates']}|")
    lines+=['', '## 参数选择','',
      f"E3/E4 共用同一 LR 策略：实际初始 LR={policy['reference_lr']}，参考batch={policy['reference_batch_size']}，alpha={policy['alpha']}，lr_factor={policy['lr_factor']}，plateau/worsening patience={policy['plateau_patience']}/{policy['worsening_patience']}。",
      '参考batch改写为32，使 reference_lr 直接等于初始batch32下的实际 LR。首个候选保留了原默认设置等价的实际 LR≈0.00659754；其前50轮指标/LR与原始记录完全一致。',
      '本轮需要调整的是衰减系数（0.5→0.7）、实际初始 LR（默认等价0.00659754→0.01）和E4缩放指数（候选中1.0最佳）。E3也记录alpha=1，以保持参数一致；B=B_ref时该指数不改变E3的LR。',
      'warmup=0、cooldown=2、min_lr=1e-8、min_delta_loss=0.001、min_delta_acc=0.002采用共同的E3入口默认设置，消除两个入口原有默认值差异；这些项未另做网格搜索。',
      'Batch控制器维持原设置：options=[32,64,128,256]，CV window=3，stability threshold=0.25，plateau patience=3，min_delta=0.01，cooldown=2。SGD momentum=0.9，weight_decay=0.0005。',
      '', '```json',json.dumps(policy,indent=2),'```',
      '', '### LR与衰减系数联合筛选（80轮）','',
      '|实际初始LR|lr_factor|验证loss|checkpoint准确率|',
      '|---:|---:|---:|---:|']
    screens=[r for r in manifest['screening_evidence'] if r['experiment']=='E3' and r['epochs']==80]
    for r in sorted(screens,key=lambda r:(r['settings']['reference_lr'],r['settings']['lr_factor'])):
        lines.append(f"|{r['settings']['reference_lr']:.8f}|{r['settings']['lr_factor']}|{r['validation_loss']:.5f}|{100*r['validation_accuracy']:.2f}%|")
    lines+=['', '### E4缩放指数筛选','',
      '|alpha|预算|验证loss|checkpoint准确率|', '|---:|---:|---:|---:|']
    screens=[r for r in manifest['screening_evidence'] if r['experiment']=='E4' and r['label'].startswith('E4_alpha')]
    for r in sorted(screens,key=lambda r:r['settings']['alpha']):
        lines.append(f"|{r['settings']['alpha']}|{r['epochs']}|{r['validation_loss']:.5f}|{100*r['validation_accuracy']:.2f}%|")
    lines+=['', '## 为什么选160轮','',
      '预算检查采用80、120、160、200轮。平台标准为末20轮的running-best验证loss改善≤0.5%；如果最后5轮刚衰减且末20轮仍改善>0.2%，延长40轮确认新台阶。',
      '80轮时优胜模型的最佳checkpoint就在第80轮；120轮时E3仍刷新最佳值，且第117轮刚发生衰减，因此延长确认。',
      f"160轮时E3/E4均满足平台条件。E3最佳checkpoint为{e3['best_epoch']}轮，E4为{e4['best_epoch']}轮；E4从120延长至160未刷新最优验证loss。因此不再延长到200。",
      '160是候选预算中经平台确认的共同展示预算，不代表最佳模型必须取第160轮。保留完整曲线，展示学习过程和后期平台。',
      '', '```json',json.dumps(params['epoch_selection'],indent=2),'```',
      '', '## E4是否更快','',
      f"相同{epochs}轮预算下，E4总训练耗时比E3少{reduction:.1f}%，optimizer updates少{update_reduction:.1f}%。这表示计算效率收益；比较收敛速度还需固定精度目标。",
      '', '|验证准确率目标|E3首次epoch|E4首次epoch|E3首次秒数|E4首次秒数|E3连续5轮确认秒数|E4连续5轮确认秒数|',
      '|---:|---:|---:|---:|---:|---:|---:|']
    for target in (.35,.40,.42,.44):
        a,b=milestone('E3',target),milestone('E4',target)
        def value(row,key):
            text=row[key]
            return ('—' if not text else f'{float(text):.2f}' if 'seconds' in key else text)
        lines.append(f"|{100*target:.0f}%|{value(a,'first_epoch')}|{value(b,'first_epoch')}|{value(a,'first_seconds')}|{value(b,'first_seconds')}|{value(a,'sustained5_confirmed_seconds')}|{value(b,'sustained5_confirmed_seconds')}|")
    a,b=milestone('E3',.42),milestone('E4',.42)
    if a['first_seconds'] and b['first_seconds']:
        speed=100*(1-float(b['first_seconds'])/float(a['first_seconds']))
        lines += ['',f"达到42%时，E3用{a['first_epoch']}轮，E4用{b['first_epoch']}轮：按epoch看E3更早；按实测耗时看E4少{speed:.1f}%。时间轴曲线解释了这两个观察为什么可以同时成立。"]
    if 'E3' in tests and 'E4' in tests:
        lines += [f"最终测试准确率E3={100*tests['E3']['accuracy']:.2f}%，E4={100*tests['E4']['accuracy']:.2f}%。E4存在耗时收益，但最终精度没有超过E3；不应把更少秒数解释为更高精度。"]
    lines += ['', '## 图片与脚本','',
      '所有loss/accuracy均单独保存epoch和真实时间两个版本，每张同时导出PNG与PDF；另保留用户喜欢的双面板布局和batch右轴。曲线不平滑、不补齐、不外推。',
      'E3固定蓝色、E4固定橙色；E1灰色、E2紫色，在独立图和四组图中一致。',
      '', '|图|epoch版本|时间版本|', '|---|---|---|']
    for label,name in [('E3/E4双面板','training_curves'),('E3/E4 loss','loss'),('E3/E4 accuracy','accuracy')]:
        first='E3_E4_training_curves.png' if name=='training_curves' else f'E3_E4_{name}_epoch.png'
        second='E3_E4_training_curves_time.png' if name=='training_curves' else f'E3_E4_{name}_time.png'
        lines.append(f'|{label}|[{first}]({root/"figures"/first})|[{second}]({root/"figures"/second})|')
    lines += ['', '每组完整结果在best_results/E1–E4。仅保留每组一份最终预算结果、最佳checkpoint及参数；本轮候选模型和中间运行目录已删除；此前实验保留。manifest.json保留小型筛选证据，verification.json记录检查与checkpoint哈希。',
      '', '复现图、最佳策略和冻结后的测试评估：','', '```sh',
      'LD_LIBRARY_PATH=/home/ams098z/miniforge3/envs/py312/lib /home/ams098z/miniforge3/envs/py312/bin/python -m experiments.plot_comparison --results-root '+str(root),
      'LD_LIBRARY_PATH=/home/ams098z/miniforge3/envs/py312/lib /home/ams098z/miniforge3/envs/py312/bin/python -m experiments.reproduce_tuned --parameters '+str(root/'best_parameters.json')+' --experiment E3',
      'LD_LIBRARY_PATH=/home/ams098z/miniforge3/envs/py312/lib /home/ams098z/miniforge3/envs/py312/bin/python -m experiments.evaluate_selected --results-root '+str(root),
      '/home/ams098z/miniforge3/envs/py312/bin/python -m experiments.summarize_selected --results-root '+str(root),
      '```','', '最佳仅指本次测试的候选范围和seed42；参数、epoch预算与checkpoint均按验证集选择，未按测试成绩重新调整。','']
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root',type=Path,required=True)
    args=parser.parse_args()
    path=args.results_root/'REPORT.md'
    path.write_text(build_report(args.results_root),encoding='utf-8')
    print(path)


if __name__=='__main__':
    main()
