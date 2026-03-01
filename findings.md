# 研究与发现记录

## 当前探索方向

- 极简 STS 语义相似度计算的底层模型架构。
- 适合教学与可视化的训练监控方法。

## 发现与结论

_(在此记录关于架构、数据集和 API 设计的启发点)_

- **语料库选择**：决定选用开源中文相似度数据集（如 LCQMC 的一个小部分子集）作为训练数据，而非手工构建极小语料。这样可以基于更真实的语言分布验证学习效果。
- **CNN 与 MeanPooling 对比实验 (2026-02-26)**：在同一 2w 数据集、同参数下，MeanPooling 极简双塔的 Accuracy 和两极分化效果**优于** CNN 双塔。
  - **根因分析**：当前使用字符级分词，CNN 的卷积核捕获的是单字组合 ("打","篮")，而非词级搭配 ("打篮球","运动")。字符级 N-gram 语义信号太弱，CNN 的局部特征提取能力被浪费。
  - **额外因素**：CNN 参数量更大容易过拟合；输出维度 192 > 128 加剧余弦相似度的维度诅咒。
  - **结论**：模型架构与分词粒度必须匹配。CNN 需要词级分词才能发挥优势。
  - **解决方案**：引入 jieba 词级分词器，CNN 选用词级分词，MeanPooling 保留字符级作为对比基线。

## 待探索的问题

1. 词量增长下的存储与加载速度优化机制。
2. 检索全量 24w 数据时的向量库构建方案（暴力计算之外的近似最近邻方案如 FAISS）。

---

## 🗄️ 上下文压缩与状态归档 (Context Compaction for Session Recovery)

_(此部分用于对话 /clear 后供大模型快速回溯恢复整周期的开发状态)_

**1. 项目核心资产 (当前状态: 阶段七全量集成与三塔成型完结)**

- `app.py`: Streamlit 网页入口，已完成前端数据量自适应选择（支持 lcqmc_mini/2w/max）、带有参数专家推荐系统。包含训练推演、折线图刷新阻断防卡死、缓存批处理的高速直方图计算。
- `model.py`: 存放三种强力底层基础组装模型：
  - `SimpleDualEncoder`: 极简基线，字嵌入 + LayerNorm + 均值池化 + Linear。
  - `CNNDualEncoder`: N-gram 抽取，词嵌入 + LayerNorm + `(2,3,4)`核维度并联池化 + BN + Dropout + Linear。
  - `LSTMDualEncoder`: 时序王者，词嵌入 + LayerNorm + 2层双向LSTM + Self-Attention池化 + Dropout + Linear。
- `train.py` & `data.py`: `data.py` 使用 jieba 为 CNN/LSTM 拆词，为 Mean 拆字，`train.py` 内含有防梯度爆炸的 `clip_grad_norm_` （LSTM 强依赖）。

**2. 最关键的踩坑记忆 (Lessons Learned)**

- 【表示坍塌】：试图在投影层绑 `tanh` 会导致大 LR 下全量激活值饱和在 ±1，余弦距离无法发散，完全重合。须保持纯线性映射。
- 【维稳机制】：1.5w词级的字典在 24w 数据下极为稀疏。如果不把所有架构的 Embedding 产出立刻经过 `LayerNorm`，不仅收敛慢，LSTM的隐层极易崩溃拉低准确度。
- 【BPTT之怒】：序列模型 (LSTM) 对高学习率极其敏感 (LR=0.3 时必为 NaN)，因为沿时间的反向传播连乘带来了梯度指数级暴涨，而 `clip_grad_norm_` 是唯一的解药。
- 【全量评测】：对于文本匹配类 STS 任务，Attention-LSTM > CNN (N-gram) > WordBag (Mean pooling)。但也更难以调参收敛。

_(如当前模型已被清空历史，请直接阅读并吸收这部分记忆恢复工作能力。)_
