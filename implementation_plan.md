# NLP Embedding 学习项目实施计划

该计划概述了从头开始构建和可视化一个极其简单的自然语言处理 (NLP) 语义 Embedding 模型的步骤，旨在用于教学和学习目的。

## 项目目标
构建一个基于 PyTorch 的极简双塔 (Dual-Encoder) 模型，用于学习句子之间的语义相似度。最终将通过 Streamlit 提供一个交互式 Web UI 界面，直观地展示训练过程 (Loss 折线图、相似度变化) 及其最终的预测与可视化降维效果。

## User Review Required
> [!NOTE]
> 这是一个基于 PyTorch 和 Streamlit 的全栈基础教学项目。
> 以下是我们拟定的极简模型架构设定，请确认是否符合“从0开始学习”的期待：
> 1. **数据**：我们将使用开源的中文相似度数据集（如 LCQMC 的一个小部分子集），以提供更真实和丰富的语料供模型学习。
> 2. **分词 (Tokenization)**：手写极其简单的基于字符级别 (Char-level) 的切词，构建字典映射，不依赖外部 `transformers` 库自带的复杂 tokenizer。
> 3. **网络结构**：单层 Embedding -> 平均池化 (Mean Pooling) -> Cosine 相似度 -> 配合简单的 Contrastive/MSE Loss 进行学习。无 Transformer 等复杂机制，最赤裸裸地展现“Embedding 是怎么被拉近推远的”。
> 4. **部署层**：基于 Streamlit 构建 `app.py` 囊括训练监控看板以及推理测试入口。

### 核心引擎 (Model & Data)
实现数据加载、模型定义和训练逻辑。考虑到教学属性，所有相关代码保持最简和高内聚。

#### [NEW] `data.py`
实现数据集的生成和处理。
*   编写代码下载或加载一个开源中文相似度数据集（如 LCQMC）的小型子集。
*   实现 `SimpleCharTokenizer` 类，包含构建词表、字符转 ID序列的功能。
*   实现 PyTorch `Dataset` 和 `DataLoader` 封装。

#### [NEW] `model.py`
构建极简的双塔模型。
*   实现 `SimpleDualEncoder` 类 (继承自 `nn.Module`)。
*   仅包含 `nn.Embedding` 层用于词嵌入，以及一个平均计算逻辑 (Mean Pooling) 得到句子表示向量。
*   实现前向传播：接收两个句子的 ID 序列，返回它们的相似度得分。

#### [NEW] `train.py`
封装训练过程，方便外部调用并在 Streamlit 中可视化。
*   定义训练循环，支持基于设定的 Epoch 步进训练。
*   支持通过 Callback (回调函数) 机制将每个 Batch / Epoch 的 Loss 暴露出来供 UI 渲染。

### 交互界面 (Web UI)
使用 Streamlit 将上述流程直观呈现出来，方便用户进行从 0 体验。

#### [NEW] `app.py`
Streamlit 主程序，包含以下侧边栏和主界面 Tabs:
*   **侧边栏**：超参数调节（Epochs, Learning Rate 等）以及“开始训练”按钮。
*   **Tab 1 (训练监控)**：展示动态折线图（Loss 的下降曲线）。
*   **Tab 2 (词汇空间降维)**：使用 PCA 或 t-SNE 将选定的词汇/短句 Embedding 降级到 2D 并在散点图上展示（直观看到相似词在空间中靠齐）。
*   **Tab 3 (预测交互)**：提供两个文本输入框，用户输入测试句子，实时计算并显示预测的相似度得分（支持后续演进为简单向量检索界面）。

#### [NEW] `requirements.txt`
包含所需的第三方依赖。
*   `torch`
*   `streamlit`
*   `pandas`
*   `numpy`
*   `matplotlib` / `scikit-learn` (用于降维画图)

### 文档规范 (Documentation)
为了保证项目的可读性和持续迭代，建立每次开发增强均伴随文档产出的机制。

#### [NEW] `docs/` 开发增强文档
*   **开发文档生成**：在每次进行新功能开发或模型增强后，均需编写/更新对应的 Markdown 开发文档（存放在 `docs/` 目录下）。
*   **文档要求**：每次文档需详细阐述此次增强的背景和目标、核心实现逻辑与设计思考、相关的 API 或接口变动、以及后续如何使用。确保对二次开发和初学者友好。

## Verification Plan

### Automated Tests
*   **基本验证脚本**：在真正接入 UI 前，我们将编写一个 `test_core.py` (不加入上面的列表中，仅作为验证手段) 快速运行一次 Dummy 数据前向传播和单步梯度下降，确保所有 Tensor 的形状匹配并没有基础 TypeError。命令：`python test_core.py`。
*   **依赖安装验证**：执行 `pip install -r requirements.txt` 确保依赖无冲突。

### Manual Verification
此项目的最终形态是完整的交互式学习。验证将以此为主：
1. 启动服务：`streamlit run app.py`。
2. 打开浏览器界面，点击 "开始训练" 按钮。
3. **验证 1（训练监控）**：观察 Loss 折线图是否随 Epochs 增加呈现下降趋势。
4. **验证 2（推理打分）**：在预测 Tab 中，输入“我也喜欢看电影”和“他真的很爱看电影”，期望给出一个较高的正向值；输入“今天天气不错”与“苹果真好吃”，期望打分明显变低甚至趋向于负数/0。
5. **验证 3（空间可视化）**：观察空间散点分布，确保意义相近的词/句子在坐标图上存在簇类聚集趋势。
