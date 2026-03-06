# NLP Embedding v2: 拥抱预训练大模型与极简 LoRA 微调

## 1. 为什么我们需要引入预训练大模型？

在第一阶段的教程中（详见 `docs/v1_nlp_embedding_tutorial.md`），我们从零开始完全手工搭建了整个 NLP 管道：

- **自研分词器**：实现了 `SimpleCharTokenizer`，手动建立由汉字到 ID 的字典映射。
- **自研模型架构**：实现了 `MeanPooling`（极简双塔）、`CNN`（局部特征双塔）、`LSTM`（时序记忆双塔），甚至纯手写的 `Transformer Dual Encoder`。

**手工造轮子的局限性**：

1. **词表太小，缺乏泛化**：单靠项目自带的 LCQMC 数据集去生成词表，无法覆盖中文庞大的生僻字词。遇到未见过的组合，模型极易坍塌。
2. **缺乏“通识语感”**：即使加了再多层级的网络架构，由于初始化是纯随机（Random Init），模型在几十万条数据的冲洗下依然只能学会表层的“字面重合度（Lexical Overlap）”。
3. **冷启动导致梯度崩溃**：如果没有经过海量语料库的预热（Warmup），从零训练复杂的 Transformer 需要极其苛刻的超参（如 Pre-LN 与严格控制的方差），小小的学习率抖动都会致使算出的相似度疯狂趋近于恒等值（Cosine Sim $\approx$ 1.0）。

为了让模型真正具备“人类级别”的文本语义理解能力，现代 NLP 的标准范式是：**Pre-train + Fine-Tune（预训练 + 微调）**。
在这个增量开发中，我们选择直接拉取国内最成熟的开源底座模型之一：`hfl/chinese-roberta-wwm-ext`（哈工大中文全词掩码预训练模型）。

- **它懂得中文**：它在超 100GB 的中文百科、新闻、问答语料库上进行了数月的训练。
- **它认识词组**：通过全词掩码（Whole Word Masking），它能把“哈尔滨”当做一个整体概念，而不是孤立的三个字。
- **架构数据**：12 层 Transformer Encoder，768 维隐层，总参数量逼近 1.03 亿。

---

## 2. 工程难题：如何优雅地把1亿参数塞进极简教程？

我们的核心哲学是 **KISS (Keep It Simple, Stupid)** 与 **白盒化**。我们不希望为了引入预训练基座，就把系统变成一个黑盒调包侠。因此，我们把改动极其精妙地揉进了原有的纯 PyTorch 引擎。

### 2.1 依赖管线的平滑切换

在 `DualEncoderEngine` 中，只要选择了包含预训练架构的模型，引擎就会自动把手工切字分词器替换为强大的 HuggingFace BPE 引擎：

```python
if self.is_pretrained:
    from transformers import AutoTokenizer
    self.tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
```

这意味着预训练模型和手工模型**互不干扰地共享着一套底层的训练循环**。

### 2.2 数据加载的效率灾难与“破局方案”

**灾难现象**：
在第一版代码中，我们每次循环到某条数据，就在 `__getitem__` 里调用切字函数把它变成 Tensor。但在 HuggingFace 体系下，`AutoTokenizer` 逐条调用的开销奇高。面对 24 万条句子的全量集（LCQMC_Max），这会导致每个 Batch 的拉取时间从几毫秒飙升到数百毫秒，GPU 会一直空转等 CPU 慢吞吞地截取文本。

**破局思路：一次性高并发预处理 + 落盘缓存**
在 `data.py` 中的 `get_pretrained_dataloader` 函数里，我们在模型开训前，采用了一种工业界标准的离线静态化策略：

1. 一次性截取所有 24 万条左句和右句。
2. 调用底层用 Rust 写成的高效 Tokenizer 批量编码数组：`enc1 = tokenizer(s1_list, return_tensors='pt')`。
3. 把生成好的几万个厚甸甸的 Tensor 全部保存进 `PretrainedSTSDataset` 对象。
4. 将该对象进行 MD5 哈希定版，并用 `pickle` 直接写入 `.pkl` 长效文件缓存。

---

## 3. 微调双营：冻结部分层(B方案) vs LoRA低秩适应

我们需要限制更新参数的范围。在这个版本中，我们实现了两种主流的微调策略，且均能在前端一键切换进行实战比对。

### 打法一：解冻顶部层 (Layer Freezing - B方案)

**思路**：语言模型的前几层负责提炼浅层语法，后几层负责高阶语义。我们的判别器是比对高阶语义相似度，所以我们只需训练顶部参数即可。
**代码实现**在 `PretrainedDualEncoder` 类中：

```python
# 冻结底座 Embedding 和 前 8 层 Transformer Encoder
self._freeze_base_layers() # 底部 71% 的参数锁死
```

**资源代价**：仅放开顶部 4 层加上我们接上的 Attention + 投影层，**依然有约 2963 万参数需要被训练**。

### 打法二：LoRA 低秩适应微调 (Low-Rank Adaptation)

这是大模型时代最为璀璨的一颗工程明珠。它利用了这样一个核心洞察：**预训练模型原本就已经懂得一切，在下游任务中，它的参数更新矩阵在数学上具有极低的“内在秩” (Intrinsic Rank)**。
对于动辄 $768 \times 768$ 尺寸的注意力权重矩阵 $W$，我们如果对其满血微调，需要更新几十万个数字。

**LoRA 的降维魔法**：

1. 原矩阵完全“焊死” ($\Delta W = 0$)，即直接冻结底座全部的 $1$ 亿参数。
2. 在这个巨大的旧矩阵旁边，我们修两条极窄的栈道相乘连起来，称之为矩阵 $A$ 和 矩阵 $B$。
3. 前向传播时，$h = Wx + BAx$。（原路信息 + 旁路补偿学习信息）。

我们在 `LoRADualEncoder` 下使用了 `peft` 库实现了这一功能。
**资源代价**：由于秩只有 8，加上我们尾部的池化与线性投影层，我们仅需训练的**参数骤减至 28,750 个**！这是之前的 **1%** 不到！

---

## 4. 403 Forbidden 与网络稳定性修复

在大模型开发中，由于权重包体积巨大（400MB+）且主要存放在境外服务器，国内开发者常遇到 403 Forbidden 或连接超时问题。为此我们实施了两项**稳定性增强**策略：

1. **镜像预注入**：在 `app.py` 启动的最顶层，优先于任何 Library 加载，注入 `HF_ENDPOINT = https://hf-mirror.com`，确保整个导入链条均使用镜像加速。
2. **“本地优先”加载策略**：在 `model.py` 中封装了 `_load_roberta` 工具函数：
   ```python
   def _load_roberta(model_name):
       try:
           # 尝试 local_files_only，如果缓存有权重则零网络请求，绝无 403
           return AutoModel.from_pretrained(model_name, local_files_only=True)
       except OSError:
           # 本地确实没有才 fallback 到正常通过镜像站下载
           return AutoModel.from_pretrained(model_name)
   ```
   这种“离线优先”的设计思路，能显著提升用户在重复切换模型时的响应速度，并规避由于镜像站瞬时波动导致的报错。

---

## 5. 工业级五大训练加速优化

为了让 1 亿参数的模型在个人电脑上也能飞速迭代，我们在 `engine.py` 和 `data.py` 中引入了五项 PyTorch 工业级加速配置：

### 5.1 AMP 自动混合精度 (FP16)

在 `_train_pretrained` 中采用 `torch.amp.autocast`。模型参数在计算时临时降维至 FP16，但梯度保持在 FP32。这能**节约 40% 以上的显存**，并让训练速度提升 **30-50%**。

### 5.2 锁页内存与预取 DataLoader (pin_memory + multi-workers)

```python
DataLoader(..., pin_memory=True, num_workers=2, persistent_workers=True)
```

- `pin_memory=True`：将 CPU 内存数据加速拷贝到 GPU。
- `num_workers=2`：开启 2 个后台子进程异步读取数据。当 GPU 正在计算当前 Batch 时，CPU 已在悄悄准备下一个 Batch。

### 5.3 异步数据传输 (non_blocking=True)

在将数据搬运到显卡时使用异步模式：

```python
label = label.to(self.device, non_blocking=True)
```

这允许 CPU 不需要等“搬运数据”这个物理过程真正结束，就可以立刻开始执行后续逻辑，让计算与搬运在时间線上重叠。

### 5.4 梯度置 None 优化 (set_to_none=True)

```python
optimizer.zero_grad(set_to_none=True)
```

传统的 `zero_grad()` 会为所有梯度填充 0 占位符。设为 `None` 则直接释放内存，减少了一次对显存的写入操作，显著提升了大型网络的反向传播效率。

### 5.5 cuDNN 自动内核算法调优 (benchmark=True)

```python
torch.backends.cudnn.benchmark = True
```

开启后，针对固定的 Input Size，cuDNN 会在内部为该卷积或矩阵乘法算子尝试多种执行策略。它会找到最快的一个并缓存。除首轮稍慢外，后续所有轮次都将运行在最优硬件链路上。

---

## 6. 实验对比与结论

| 对比维度                  | 冻结底部 8 层 (方案 B)                 | LoRA 小口径微调 (r=8)                        |
| :------------------------ | :------------------------------------- | :------------------------------------------- |
| **可训练参数量**          | 29,630,000 (约 28.8%)                  | 28,750 (约 0.03%)                            |
| **显存使用率**            | 较高（~3.2GB+）                        | 极低（~1.8GB+）                              |
| **加速后的单 Epoch 耗时** | 显著下降 (得益于 AMP)                  | 极速响应                                     |
| **性能极限**              | 上限极高，适合定制化垂直领域深度微调。 | 极易训练，不易过拟合。在数据量少时表现惊人。 |

当你熟练掌握了这个架构，你就能用一台普通 8GB 显存显卡的家用电脑，极其轻易地将包含数十亿甚至数千亿参数的开源 LLM 系统地引入到类似的降秩轨道，真正敲开大模型微调世界的殿堂大门。
