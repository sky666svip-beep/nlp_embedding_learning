# 阶段：预训练模型高级微调策略 (LR Scheduler & Early Stopping)

## 目标

- [ ] 在 `engine.py` 的 `_train_pretrained` 分支中引入学习率调度器 (Linear Warmup + Cosine Annealing)。
- [ ] 在训练中加入 Early Stopping 机制以避免过拟合浪费算力，当连续 N 轮 (patience) Validation/Train Loss 无法下降时自动拉停。
- [ ] 确保与现有的 AMP 混合精度训练及回调前端可视化的逻辑完美结合。

## 详细实施计划

1. 修改 `engine.py` 中的 `_train_pretrained` 方法。
2. 引入 `torch.optim.lr_scheduler.LinearLR` (做前 N % 步数的 Warmup) 以及 `torch.optim.lr_scheduler.CosineAnnealingLR` (做剩余步数的余弦退火)，使用 `SequentialLR` 串联。
3. 实现 Early Stopping 逻辑 (检测 epoch loss 并在触发 patience 时 `break`)。
4. 更新 progress 与 findings。
