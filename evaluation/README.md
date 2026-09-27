# Evaluation 接入说明

本目录只负责评估，不启动训练，也不修改 optimizer。所有 accuracy、precision、recall、F1 输入/输出使用 **0–1**；展示百分比时乘以 100。时间单位为秒。泛化差距单独以**百分点**返回。

## 文件与功能

- `metrics.py`：测试 loss、accuracy、macro precision/recall/F1、各类别指标、训练计时、目标达成时间、更新次数、泛化差距。
- `confusion_matrix.py`：保存原始计数或按真实类别归一化的混淆矩阵。
- `plots.py`：多个策略的 train/validation loss 和 accuracy，分别以 epoch 和累计时间为横轴。

从项目根目录导入这些模块，不要直接运行本目录里的文件。

## 测试集评估

```python
import json
from pathlib import Path
from evaluation import evaluate_model, generalization_gap
from evaluation.confusion_matrix import plot_confusion_matrix

# model 已加载按验证集选出的 checkpoint，并已放到 device。
result = evaluate_model(model, data.get_test_loader(), device)
print(f"Test accuracy: {result['accuracy']:.2%}")
print(f"Macro F1: {result['macro_f1']:.4f}")

output = Path("results") / "adaptive_lr"
output.mkdir(parents=True, exist_ok=True)
(output / "test_metrics.json").write_text(
    json.dumps(result, indent=2), encoding="utf-8",
)
plot_confusion_matrix(
    result["y_true"], result["y_pred"], output / "confusion_matrix.png",
)
plot_confusion_matrix(
    result["y_true"], result["y_pred"], output / "confusion_matrix_normalized.png",
    normalize=True,
)
```

`evaluate_model` 使用无类别权重的交叉熵，按实际样本数平均，支持最后一个不足 batch size 的批次。它自动使用 eval/no_grad，并恢复调用前各子模块的 train/eval 状态。空数据集和非有限 loss 会报错。分类指标固定包含全部十类，无法定义的 precision/recall/F1 记为 0。

## 训练历史接口（交给 Dev 接入）

每个 epoch 验证结束后追加一个字典：

```python
history.append({
    "epoch": epoch,                         # 从 1 开始
    "train_loss": train_loss,               # 按样本平均
    "val_loss": val_loss,                   # 按样本平均
    "train_acc": train_accuracy,            # 0–1
    "val_acc": val_accuracy,                # 0–1
    "elapsed_seconds": timer.elapsed_seconds,  # 从训练开始累计
    "optimizer_updates": epoch_updates,     # 仅本轮实际 optimizer.step 次数
})
```

`optimizer_updates` 必须在真正执行参数更新时累加，不能直接拿 batch 数或 `ceil(N / batch_size)` 代替。梯度累积和 AMP 跳过更新时尤其要注意。该字段为每轮计数，汇总函数将其相加。

计时器使用方式如下（训练和验证部分由公共训练器提供）：

```python
from evaluation import TrainingTimer, summarize_training
from evaluation.plots import plot_training_curves

timer = TrainingTimer(device)
history = []
with timer.measure():
    # 在这里执行完整 epoch 循环：训练、验证、追加上述 history 行。
    # 验证完成后读取 timer.elapsed_seconds。
    ...

# history 填充后执行：
summary = summarize_training(
    history,
    total_training_seconds=timer.total_seconds,
    target_accuracy=0.80,
)
plot_training_curves({"adaptive_lr": history}, "results/adaptive_lr/curves.png")
```

这是集成模板，不是独立训练脚本。计时包含循环内的数据读取、训练、验证和其他操作；不应把下载数据、最终测试和绘图放进计时区域。所有策略保持相同计时边界。CUDA/MPS 在读取时间时同步，避免遗漏尚未完成的设备计算。

`summary` 包含：

- `training_time_seconds`：完整计时区域耗时。
- `optimizer_updates`：实际更新总次数。
- `epoch_to_target` / `time_to_target_seconds`：**首次**达到目标准确率的验证 epoch 及累计时间；从未达到时均为 `None`（JSON 中为 null）。这是 epoch 结束时的测量精度，不是批次级精度。
- `target_accuracy` / `target_split`：目标值与使用的数据集（validation）。

比较多个策略时传入 `{"fixed_fixed": history_a, "adaptive_batch": history_b, "adaptive_lr": history_c, "adaptive_combined": history_d}`。不同策略允许有不同 epoch 数。

## 泛化差距

```python
# train_eval_loader 使用原训练子集，但关闭随机裁剪、翻转等数据增强。
train_result = evaluate_model(model, train_eval_loader, device)
gap = generalization_gap(train_result["accuracy"], result["accuracy"])
print(f"Generalization gap: {gap:.2f} percentage points")
```

必须用同一个 checkpoint 对两个数据集评估。当前 `get_train_loader()` 有随机增强，不适合作为这里的公平训练评估 loader；请由数据/训练模块负责人提供无随机增强的训练子集 loader。不要使用训练过程中权重不断变化时累计的 train accuracy 计算最终泛化差距。差距为负时保留负值。
