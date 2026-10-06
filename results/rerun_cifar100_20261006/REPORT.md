# CIFAR-100 新参数实验结果

推荐共同展示预算：**160 epochs**。seed=42；训练45,000、验证5,000、官方测试10,000张；模型、数据划分、SGD和归一化保持一致。
模型按最低验证 loss 选择 checkpoint；测试集仅在参数和预算冻结后评估，不参与重新选参。

## 最终结果

|实验|预算|最佳checkpoint|验证loss|验证准确率|测试准确率|测试macro F1|训练秒数|更新次数|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|E1|160|144|2.31332|39.94%|40.03%|0.3856|656.06|225120|
|E2|160|144|2.24708|41.34%|41.49%|0.4054|448.22|54894|
|E3|160|123|2.18429|43.56%|43.41%|0.4220|662.15|225120|
|E4|160|106|2.21916|42.54%|42.11%|0.4084|466.20|70018|

## 参数选择

E3/E4 共用同一 LR 策略：实际初始 LR=0.01，参考batch=32，alpha=1.0，lr_factor=0.7，plateau/worsening patience=2/2。
参考batch改写为32，使 reference_lr 直接等于初始batch32下的实际 LR。首个候选保留了原默认设置等价的实际 LR≈0.00659754；其前50轮指标/LR与原始记录完全一致。
本轮需要调整的是衰减系数（0.5→0.7）、实际初始 LR（默认等价0.00659754→0.01）和E4缩放指数（候选中1.0最佳）。E3也记录alpha=1，以保持参数一致；B=B_ref时该指数不改变E3的LR。
warmup=0、cooldown=2、min_lr=1e-8、min_delta_loss=0.001、min_delta_acc=0.002采用共同的E3入口默认设置，消除两个入口原有默认值差异；这些项未另做网格搜索。
Batch控制器维持原设置：options=[32,64,128,256]，CV window=3，stability threshold=0.25，plateau patience=3，min_delta=0.01，cooldown=2。SGD momentum=0.9，weight_decay=0.0005。

```json
{
  "reference_batch_size": 32,
  "alpha": 1.0,
  "lr_factor": 0.7,
  "plateau_patience": 2,
  "worsening_patience": 2,
  "cooldown_epochs": 2,
  "warmup_epochs": 0,
  "min_lr": 1e-08,
  "min_delta_loss": 0.001,
  "min_delta_acc": 0.002,
  "reference_lr": 0.01
}
```

### LR与衰减系数联合筛选（80轮）

|实际初始LR|lr_factor|验证loss|checkpoint准确率|
|---:|---:|---:|---:|
|0.00659754|0.5|2.34482|39.42%|
|0.00659754|0.7|2.22079|42.24%|
|0.01000000|0.5|2.36252|39.38%|
|0.01000000|0.7|2.21283|42.82%|
|0.02000000|0.5|2.33325|39.96%|
|0.02000000|0.7|2.24032|42.52%|

### E4缩放指数筛选

|alpha|预算|验证loss|checkpoint准确率|
|---:|---:|---:|---:|
|0.0|120|2.28758|40.50%|
|0.2|120|2.26515|41.16%|
|0.5|120|2.24542|41.72%|
|1.0|120|2.21916|42.54%|

## 为什么选160轮

预算检查采用80、120、160、200轮。平台标准为末20轮的running-best验证loss改善≤0.5%；如果最后5轮刚衰减且末20轮仍改善>0.2%，延长40轮确认新台阶。
80轮时优胜模型的最佳checkpoint就在第80轮；120轮时E3仍刷新最佳值，且第117轮刚发生衰减，因此延长确认。
160轮时E3/E4均满足平台条件。E3最佳checkpoint为123轮，E4为106轮；E4从120延长至160未刷新最优验证loss。因此不再延长到200。
160是候选预算中经平台确认的共同展示预算，不代表最佳模型必须取第160轮。保留完整曲线，展示学习过程和后期平台。

```json
{
  "frontier": [
    {
      "epochs": 160,
      "E3": {
        "last20_relative_loss_improvement": 0.0,
        "last_decay_epoch": 159,
        "settled": true
      },
      "E4": {
        "last20_relative_loss_improvement": 0.0,
        "last_decay_epoch": 160,
        "settled": true
      }
    }
  ],
  "E3": {
    "last20_relative_loss_improvement": 0.0,
    "last_decay_epoch": 159,
    "settled": true
  },
  "E4": {
    "last20_relative_loss_improvement": 0.0,
    "last_decay_epoch": 160,
    "settled": true
  }
}
```

## E4是否更快

相同160轮预算下，E4总训练耗时比E3少29.6%，optimizer updates少68.9%。这表示计算效率收益；比较收敛速度还需固定精度目标。

|验证准确率目标|E3首次epoch|E4首次epoch|E3首次秒数|E4首次秒数|E3连续5轮确认秒数|E4连续5轮确认秒数|
|---:|---:|---:|---:|---:|---:|---:|
|35%|31|31|129.08|127.12|182.81|162.15|
|40%|54|56|224.68|196.87|285.34|207.19|
|42%|69|75|285.34|246.29|368.29|295.49|
|44%|—|—|—|—|—|—|

达到42%时，E3用69轮，E4用75轮：按epoch看E3更早；按实测耗时看E4少13.7%。时间轴曲线解释了这两个观察为什么可以同时成立。
最终测试准确率E3=43.41%，E4=42.11%。E4存在耗时收益，但最终精度没有超过E3；不应把更少秒数解释为更高精度。

## 图片与脚本

所有loss/accuracy均单独保存epoch和真实时间两个版本，每张同时导出PNG与PDF；另保留用户喜欢的双面板布局和batch右轴。曲线不平滑、不补齐、不外推。
E3固定蓝色、E4固定橙色；E1灰色、E2紫色，在独立图和四组图中一致。

|图|epoch版本|时间版本|
|---|---|---|
|E3/E4双面板|[E3_E4_training_curves.png](/home/ams098z/Courses/CSCE5218/MidetermProj/adaptive-batch-lr-cifar10/results/rerun_cifar100_20261006/figures/E3_E4_training_curves.png)|[E3_E4_training_curves_time.png](/home/ams098z/Courses/CSCE5218/MidetermProj/adaptive-batch-lr-cifar10/results/rerun_cifar100_20261006/figures/E3_E4_training_curves_time.png)|
|E3/E4 loss|[E3_E4_loss_epoch.png](/home/ams098z/Courses/CSCE5218/MidetermProj/adaptive-batch-lr-cifar10/results/rerun_cifar100_20261006/figures/E3_E4_loss_epoch.png)|[E3_E4_loss_time.png](/home/ams098z/Courses/CSCE5218/MidetermProj/adaptive-batch-lr-cifar10/results/rerun_cifar100_20261006/figures/E3_E4_loss_time.png)|
|E3/E4 accuracy|[E3_E4_accuracy_epoch.png](/home/ams098z/Courses/CSCE5218/MidetermProj/adaptive-batch-lr-cifar10/results/rerun_cifar100_20261006/figures/E3_E4_accuracy_epoch.png)|[E3_E4_accuracy_time.png](/home/ams098z/Courses/CSCE5218/MidetermProj/adaptive-batch-lr-cifar10/results/rerun_cifar100_20261006/figures/E3_E4_accuracy_time.png)|

每组完整结果在best_results/E1–E4。仅保留每组一份最终预算结果、最佳checkpoint及参数；本轮候选模型和中间运行目录已删除；此前实验保留。manifest.json保留小型筛选证据，verification.json记录检查与checkpoint哈希。

复现图、最佳策略和冻结后的测试评估：

```sh
LD_LIBRARY_PATH=/home/ams098z/miniforge3/envs/py312/lib /home/ams098z/miniforge3/envs/py312/bin/python -m experiments.plot_comparison --results-root /home/ams098z/Courses/CSCE5218/MidetermProj/adaptive-batch-lr-cifar10/results/rerun_cifar100_20261006
LD_LIBRARY_PATH=/home/ams098z/miniforge3/envs/py312/lib /home/ams098z/miniforge3/envs/py312/bin/python -m experiments.reproduce_tuned --parameters /home/ams098z/Courses/CSCE5218/MidetermProj/adaptive-batch-lr-cifar10/results/rerun_cifar100_20261006/best_parameters.json --experiment E3
LD_LIBRARY_PATH=/home/ams098z/miniforge3/envs/py312/lib /home/ams098z/miniforge3/envs/py312/bin/python -m experiments.evaluate_selected --results-root /home/ams098z/Courses/CSCE5218/MidetermProj/adaptive-batch-lr-cifar10/results/rerun_cifar100_20261006
/home/ams098z/miniforge3/envs/py312/bin/python -m experiments.summarize_selected --results-root /home/ams098z/Courses/CSCE5218/MidetermProj/adaptive-batch-lr-cifar10/results/rerun_cifar100_20261006
```

最佳仅指本次测试的候选范围和seed42；参数、epoch预算与checkpoint均按验证集选择，未按测试成绩重新调整。
