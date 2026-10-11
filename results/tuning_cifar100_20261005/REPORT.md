# CIFAR-100 tuning results

Common budget: 160 epochs; seed42; unchanged CNN; validation only.

Best means lowest validation loss among tested candidates, not a global optimum.

E3 searched reference LR 0.005/0.01/0.02 and worsening patience 3/5 (four combinations). E4 searched alpha 0/0.2/0.5 using the selected E3 LR policy and unchanged batch controller.

E1/E2 use selected E3 reference LR at batch32/Bref32, so their fixed LR equals E3 initial LR.

|Experiment|Epochs|Best epoch|Val loss|Val accuracy|Macro F1|Updates|Train seconds|
|---|---:|---:|---:|---:|---:|---:|---:|
|E1|160|144|2.31332|39.94%|0.3822|225120|634.48|
|E2|160|144|2.24708|41.34%|0.4012|54894|433.13|
|E3|160|127|2.11330|44.20%|0.4319|225120|685.84|
|E4|160|106|2.21134|42.32%|0.4103|55950|433.24|

Convergence diagnostics and exact settings are in manifest.json and best_parameters.json. Best checkpoints can occur at different epochs; training budgets are equal. No test-set evaluation or multi-seed confirmation was performed.

## 参数与轮数选择

固定 CNN、CIFAR-100、seed42、batch 初始32、SGD LR初始0.01。E1/E2保留原有基线的初始batch、LR和batch控制器参数；E3/E4使用调优的LR控制器。所有实验训练160轮，分别评估各自最低验证loss的checkpoint，因此最佳checkpoint的epoch可以不同。

E3在120轮时末20轮最佳loss仍改善1.67%，因此延长至160轮；160轮的末20轮没有刷新最佳loss。最佳checkpoint位于127轮，故不再延长至200轮。

|参数|E3|E4|
|---|---:|---:|
|reference_batch_size|32|32|
|alpha|0.2|0.5|
|lr_factor|0.5|0.5|
|plateau_patience|5|5|
|worsening_patience|5|5|
|cooldown_epochs|2|2|
|warmup_epochs|0|0|
|min_lr|1e-06|1e-06|
|min_delta_loss|0.001|0.001|
|min_delta_acc|0.002|0.002|
|reference_lr|0.01|0.01|

E2/E4 batch控制器：options=[32,64,128,256]，CV window=3，stability threshold=0.25，plateau patience=3，min delta=0.01，cooldown=2。E3 batch固定32，故alpha不影响其LR（B=Bref）。E4的alpha由搜索确定。

## 候选结果

|试验|轮数|验证loss|最佳checkpoint准确率|训练秒数|
|---|---:|---:|---:|---:|
|E3_lr0.005_w3_ep120|120|2.23786|42.20%|495.19|
|E3_lr0.01_w3_ep120|120|2.20428|42.82%|510.13|
|E3_lr0.02_w3_ep120|120|2.18611|43.56%|519.98|
|E3_lr0.01_w5_ep120|120|2.12648|43.82%|515.61|
|E3_final_ep160|160|2.11330|44.20%|685.84|
|E3_runnerup_ep160|160|2.18611|43.56%|631.09|
|E4_alpha0.0_ep160|160|2.26116|41.68%|434.27|
|E4_alpha0.2_ep160|160|2.27866|41.02%|434.12|
|E4_alpha0.5_ep160|160|2.21134|42.32%|433.24|
|E1_final_ep160|160|2.31332|39.94%|634.48|
|E2_final_ep160|160|2.24708|41.34%|433.13|

E4比E3训练耗时减少36.8%，最佳checkpoint准确率差1.88个百分点。这是相同epoch预算下的耗时与质量比较，不能直接称为达到相同精度更快收敛；各精度阈值的首次达到时间另存time_to_accuracy.csv。

## 文件与复现

best_results/E1–E4 各目录保留完整checkpoint、CSV、metadata、验证指标、混淆矩阵与曲线。comparison_training_curves.png为四组汇总，E3_E4_training_curves.png为重点比较。source_snapshot保留搜索时源码，manifest.json保留哈希与全部试验。

在原项目目录、原py312环境与当前源码配置下运行（将E3替换为E1/E2/E4即可）：

```sh
LD_LIBRARY_PATH=/home/ams098z/miniforge3/envs/py312/lib /home/ams098z/miniforge3/envs/py312/bin/python -m experiments.reproduce_tuned --parameters results/tuning_cifar100_20261005/best_parameters.json --experiment E3
```

“最佳”限于本次候选范围和seed42。调参只使用验证集；未使用测试集选参数，也尚未进行多seed稳定性检验。训练时间不包含绘图和最终评估开销。

## 相同终点的指标

以下均取第160轮训练日志，不与各自最佳checkpoint混淆。

|实验|epoch|验证loss|验证准确率|batch|
|---|---:|---:|---:|---:|
|E1|160|2.44736|36.94%|32|
|E2|160|2.34259|39.78%|256|
|E3|160|2.12148|44.26%|32|
|E4|160|2.21704|42.46%|256|

达到42%验证准确率：E3首次在第80轮、346.40秒；E4也在第80轮、232.00秒。E4达到相同42%阈值耗时更少，但本轮未达到44%；E3在第116轮达到44%。这些是单次越过阈值的时间，不代表此后每轮均维持该精度。
