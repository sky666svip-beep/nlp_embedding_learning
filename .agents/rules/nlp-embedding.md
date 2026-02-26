---
trigger: always_on
---

## 1. 核心开发哲学
*   **KISS (Keep It Simple, Stupid)**：本项目以“教学与直观学习”为核心目的。不引入额外的抽象、庞大的第三方类库（如 `transformers` 或 `datasets`）、复杂的调度器或海量训练数据。
*   **白盒化 (White-box)**：确保每一步计算都能被轻松打印和绘制出来。像 `SimpleCharTokenizer` 和 `SimpleDualEncoder` 一样，模型内必须保留清晰明了的数据流动逻辑。
*   **即时可视化反馈 (Immediate Visual Feedback)**：任何后端的结构演进（如加入分类层、更改Loss），能够展现时，必须在对应的 Streamlit 页面 (`app.py`) 中直观地体现出来。

## 2. Manus 风格的文件管理与任务规划
本工作区采用了 Manus 驱动的落地机制，通过记录文件（Markdown）替代易失的短期记忆。在每次进入**全新的增量开发模块**前，必须经过以下流程：

1.  **规划与检查 (`task_plan.md`)**：
    *   在开始编写代码前，确保相关目标已记录为 `.md` 中的任务 CheckBox (`- [ ]`)。
    *   按照 `[阶段N：目标名称]` 的方式来组织结构。
2.  **探索与记录 (`findings.md`)**：
    *   如果在开发过程中遇到困难、需查证的假设或灵感，必须记录在这里。
    *   *准则：绝不重复同一个错误。如果一次尝试失败，将其方法和错误记录在案后寻找新思路。*
3.  **日志与状态 (`progress.md`)**：
    *   每当结束一次对话会话，或者完成了一个功能增量（例如“接入 GPU 支持”），在此记录当期的里程碑结果，保持“当前进展”常新。

## 3. 代码与架构规范

### Python 代码规范
*   遵循基础清晰的 PEP8 命名惯例（如使用小写及下划线命名变量和函数 `get_loaded_model`，首字母大写命名类 `SimpleDualEncoder`）。
*   **类型张量一致性**：永远警惕张量所在的设备。默认写法应当类似于：
    ```python
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    tensor = tensor.to(device)
    ...
    # CPU 环境渲染回抽前需明确分离
    result = tensor.cpu().detach().numpy()
    ```

### 仓库目录职责
*   **`data/`**：仅存放轻量级的原始测试语料（如 `lcqmc_mini.csv`）。
*   **`docs/`**：每一次**跨越至少2至3个文件的重链式增强实现后**，必须产出一篇说明性文档存放于此。目的是为了留下可读的教学和复盘记录。
*   **`output/`**：只存放模型被训练后落盘的持久化文件（如 `.pth` 权重， `.pkl` 字典等），必须保持该文件夹对于核心 Git 的独立性（需配置在 `.gitignore` 内）。
*   **`核心四件套`**：
    *   `data.py`: 分词与数据加载。
    *   `model.py`: 核心网络结构。
    *   `train.py`: 学习闭包与批次循环。
    *   `app.py`: Streamlit 交互与调度主函数。

## 4. 提交 / 更新的自检清单
在输出任意“Task Complete”前，请确认：
- [ ] 代码修改是否能被成功启动 `streamlit run app.py` 而不崩溃？
- [ ] 针对界面或后端的修改，是否同步在相关的 Markdown（尤其是任务与日志文件）中勾除了待办项？
- [ ] 新功能是否破坏了极简的“双塔平均池化”这一核心教学特征？（如未经探讨，请勿私自修改其底层结构）
