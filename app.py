import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from engine import DualEncoderEngine
import torch
import os
import pickle
import matplotlib.pyplot as plt
import time
import faiss
import faiss

st.set_page_config(page_title="NLP Embedding 学习", layout="wide")
st.title("NLP Embedding 双塔模型")

# Sidebar
st.sidebar.header("[训练参数]")
model_arch = st.sidebar.selectbox("模型架构", ["MeanPooling 极简双塔", "CNN 双塔", "LSTM 双塔", "Transformer 双塔"], index=0)
model_type_map = {"MeanPooling 极简双塔": "mean_pooling", "CNN 双塔": "cnn", "LSTM 双塔": "lstm", "Transformer 双塔": "transformer"}
selected_model_type = model_type_map[model_arch]

dataset_scale = st.sidebar.selectbox("训练数据规模", [
    "全量集 (lcqmc_max, 约24w条)", 
    "中型集 (lcqmc_2w, 约2w条)", 
    "迷你集 (lcqmc_mini, 5k条)"
], index=0)

dataset_map = {
    "全量集 (lcqmc_max, 约24w条)": "data/lcqmc_max.csv",
    "中型集 (lcqmc_2w, 约2w条)": "data/lcqmc_2w.csv",
    "迷你集 (lcqmc_mini, 5k条)": "data/lcqmc_mini.csv"
}
selected_dataset_path = dataset_map[dataset_scale]

st.sidebar.markdown("---")
st.sidebar.subheader("[推荐参数设置]")
if "max" in selected_dataset_path:
    st.sidebar.info("大语料建议：Epochs: 1-3 | LR: 0.0005 | Batch Size: 128或256")
    default_epochs, default_lr, default_batch = 2, 0.0005, 3
elif "2w" in selected_dataset_path:
    st.sidebar.info("中语料建议：Epochs: 5-10 | LR: 0.001 | Batch Size: 64")
    default_epochs, default_lr, default_batch = 5, 0.001, 2
else:
    st.sidebar.info("微语料建议：Epochs: 15-30 | LR: 0.005 | Batch Size: 16")
    default_epochs, default_lr, default_batch = 15, 0.005, 0

epochs = st.sidebar.slider("Epochs", min_value=1, max_value=50, value=default_epochs)
lr = st.sidebar.number_input("Learning Rate", value=default_lr, format="%.4f")
batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64, 128, 256], index=default_batch)
embed_dim = st.sidebar.slider("词向量维度", min_value=8, max_value=256, value=128)

if "model_state" not in st.session_state:
    st.session_state.model_state = None
if "tokenizer" not in st.session_state:
    st.session_state.tokenizer = None
if "model_type" not in st.session_state:
    st.session_state.model_type = "mean_pooling"
if "loss_history" not in st.session_state:
    st.session_state.loss_history = []
if "acc_history" not in st.session_state:
    st.session_state.acc_history = []
# 用一个随机数或时间戳作为缓存失效的 key（当新训练完成时强制刷新缓存）
if "train_version" not in st.session_state:
    st.session_state.train_version = 0

tab1, tab2, tab3, tab4 = st.tabs(["1. 训练与评估监控", "2. 空间特征降维", "3. 模型预测", "4. 向量检索与应用"])

with st.sidebar:
    start_train = st.button("开始训练", use_container_width=True)
    
    st.divider()
    if st.session_state.model_state is not None:
        if st.button("保存模型到本地", use_container_width=True):
            os.makedirs("output", exist_ok=True)
            # Save weights
            torch.save(st.session_state.model_state, "output/model_weights.pth")
            # Save tokenizer vocabulary simply using pickle
            with open("output/tokenizer.pkl", "wb") as f:
                pickle.dump(st.session_state.tokenizer, f)
            st.success("模型权重与词表已保存至 `output/` 目录！")

if start_train:
    st.session_state.loss_history = []
    st.session_state.acc_history = []
    st.session_state.train_version += 1  # 训练版本加一，让旧图表缓存失效
    
    with tab1:
        st.subheader("模型训练与收敛状态")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        engine = DualEncoderEngine(selected_model_type, embed_dim=embed_dim)
        
        # 计算总 batch 数以设置进度条
        from data import get_dataloader
        mock_dl, mock_tk = get_dataloader(selected_dataset_path, batch_size=batch_size, tokenizer_type=engine.tok_type)
        total_batches = len(mock_dl)
        total_steps = epochs * total_batches
        
        def train_callback(epoch, batch_idx, loss, batch_acc):
            st.session_state.loss_history.append(loss)
            st.session_state.acc_history.append(batch_acc)
            current_step = epoch * total_batches + batch_idx + 1
            progress_bar.progress(current_step / total_steps)
            # 仅在每个 epoch 结束时更新状态文本
            if batch_idx == total_batches - 1:
                status_text.text(f"Epoch {epoch+1}/{epochs} 完成 | 平均Loss: {loss:.4f} | 末尾Acc: {batch_acc:.4f}")
            
        with st.spinner(f"模型 ({model_arch}) 正在学习语义分布中..."):
            engine.train(selected_dataset_path, epochs=epochs, batch_size=batch_size, lr=lr, callback=train_callback)
            st.session_state.model_state = engine.model.state_dict()
            st.session_state.tokenizer = engine.tokenizer
            st.session_state.model_type = selected_model_type
            
        # 训练结束后一次性渲染最终曲线
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.write("[Loss 曲线]")
            st.line_chart(st.session_state.loss_history)
        with col_c2:
            st.write("[Accuracy 曲线]")
            st.line_chart(st.session_state.acc_history)
        st.success(f"训练完成（{model_arch}）！请前往其他 Tab 查看效果。")
else:
    with tab1:
        st.subheader("模型训练与收敛状态")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.write("[Loss (均方误差) 曲线]")
            if st.session_state.loss_history:
                st.line_chart(st.session_state.loss_history)
            else:
                st.info("等待数据。")
        with col_c2:
            st.write("[Accuracy (批次准确率) 曲线]")
            if st.session_state.acc_history:
                st.line_chart(st.session_state.acc_history)
            else:
                st.info("等待数据。")
        
        if st.session_state.loss_history:
            st.success("这是当前存留的模型训练走势。")
        else:
            st.info("请先在侧边栏点击【开始训练】。")

@st.cache_resource
def get_cached_engine(model_type, model_state_bytes, _tokenizer):
    """缓存加载的模型引擎，避免每次重新构建和加载权重。"""
    import io
    buffer = io.BytesIO(model_state_bytes)
    state = torch.load(buffer, weights_only=True)
    engine = DualEncoderEngine(model_type, embed_dim=embed_dim, tokenizer=_tokenizer, model_state=state)
    return engine

def get_loaded_engine():
    if not st.session_state.model_state or not st.session_state.tokenizer:
        return None
    import io
    buffer = io.BytesIO()
    torch.save(st.session_state.model_state, buffer)
    engine = get_cached_engine(st.session_state.model_type, buffer.getvalue(), st.session_state.tokenizer)
    return engine

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

@st.cache_data
def compute_similarity_distribution(version, _engine, dataset_path):
    """计算数据集的全局相似度分布，通过 version 参数控制刷新"""
    df = pd.read_csv(dataset_path)
    
    # 使用统一推理引擎
    s1_list = df['sentence1'].tolist()
    s2_list = df['sentence2'].tolist()
    labels = df['label'].values
    
    sim_vals = _engine.predict_similarity(s1_list, s2_list, batch_size=256)
    
    pos_sims = sim_vals[labels == 1].tolist()
    neg_sims = sim_vals[labels == 0].tolist()
    
    return pos_sims, neg_sims

with tab1:
    st.divider()
    st.subheader("[全局预测分布透视图] (相似度分布直方图)")
    if st.session_state.model_state:
        engine = get_loaded_engine()
        
        with st.spinner("计算全局相似度分布 (已利用 GPU 批处理提速 & 数据缓存)..."):
            pos_sims, neg_sims = compute_similarity_distribution(st.session_state.train_version, engine, selected_dataset_path)
            
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.hist(pos_sims, bins=np.linspace(-1, 1, 21), alpha=0.6, label='Similar (Label=1)', color='green')
            ax.hist(neg_sims, bins=np.linspace(-1, 1, 21), alpha=0.6, label='Dissimilar (Label=0)', color='red')
            ax.set_title('Cosine Similarity Distribution Analysis')
            ax.set_xlabel('Cosine Similarity Score')
            ax.set_ylabel('Frequency')
            ax.legend(loc='upper right')
            st.pyplot(fig)
            st.markdown("**怎么看这张图？** 绿色直方图越靠右 (趋近 1)，红色直方图越靠左 (趋近 -1)，重叠部分越少，说明模型“区分句子相似与否的能力”越强（两极分化越好）！")
    else:
        st.info("尚未完成训练，无法查看分布直方图。")

@st.cache_data
def get_pca_data(version, _engine, sentences):
    """缓存 PCA 降维坐标"""
    embs = _engine.encode(sentences, batch_size=256)
    if len(embs) > 0:
        coords = PCA(n_components=2).fit_transform(embs)
        return coords
    return np.array([])

with tab2:
    st.subheader("二维句子空间分布 (PCA降维)")
    st.markdown("将高维的句子向量压缩至2D平面，距离相近的点代表模型认为它们语义相似。")
    if st.session_state.model_state:
        engine = get_loaded_engine()
        df = pd.read_csv(selected_dataset_path)
        # 提取前 30 对句子用于展示
        sentences = list(set(df['sentence1'].tolist()[:30] + df['sentence2'].tolist()[:30]))
        
        with st.spinner("抽取高维特征并计算 PCA..."):
            coords = get_pca_data(st.session_state.train_version, engine, sentences)
            chart_df = pd.DataFrame({"X": coords[:, 0], "Y": coords[:, 1], "Text": sentences})
            
            # Use columns to lay out side-by-side
            c1, c2 = st.columns([2, 1])
            with c1:
                st.scatter_chart(chart_df, x="X", y="Y")
            with c2:
                st.dataframe(chart_df[["X", "Y", "Text"]], use_container_width=True)
                st.caption("左侧图表为点位分布，对照此表可查找具体句子所在坐标。")
    else:
        st.info("请先在侧边栏点击【开始训练】。")

with tab3:
    st.subheader("输入两句话，预测相似度指数")
    engine = get_loaded_engine()
    
    col1, col2 = st.columns(2)
    s1 = col1.text_input("第一句话", "苹果手机怎么截图")
    s2 = col2.text_input("第二句话", "iPhone屏显怎么截")
        
    if st.button("计算相似度", type="primary"):
        if engine:
            sims = engine.predict_similarity([s1], [s2])
            similarity_score = sims[0]
            
            st.metric(label="Cosine Similarity (余弦相似度)", value=f"{similarity_score:.4f}")
            if similarity_score > 0.5:
                st.success("判断：它们大概率是 **相似** 的！")
            else:
                st.warning("判断：它们可能 **不相似**。")
        else:
            st.error("出错： 模型未初始化，请先训练！")

with tab4:
    st.subheader("构建本地微型向量库与加速检索体验")
    st.markdown("将全量数据集的句子映射成特征向量存入“向量库”，体验通过一句话瞬间找回最相似的 Top-K 条语料。通过对比普通「PyTorch 暴力矩阵算分」与大厂使用的「FAISS 近似检索引擎」，直观感受性能差异。")
    
    col_v1, col_v2 = st.columns([1, 2])
    with col_v1:
        st.info("提示：必须先训练或加载模型，才能将其能力用于构建向量库。")
        build_db_btn = st.button("根据当前语料库构建全量向量索引", type="primary", use_container_width=True)
    
    if "vector_db" not in st.session_state:
        st.session_state.vector_db = None
    
    if build_db_btn:
        if not st.session_state.model_state:
            st.error("出错： 当前无可用模型！请先在侧拉栏点击【开始训练】。")
        else:
            engine = get_loaded_engine()
            df = pd.read_csv(selected_dataset_path)
            
            with st.spinner("1/3 正在提取唯一句子字典 (过滤重复语句)..."):
                # 为了防止两两配对数据里大量重复句子占用显存，首先去重
                s1_unique = df[['sentence1', 'label']].rename(columns={'sentence1': 'text'})
                s2_unique = df[['sentence2', 'label']].rename(columns={'sentence2': 'text'})
                # 由于相同的句子如果在原表中遇到过正样本又遇到负样本，保留第一次出现的，仅作为检索结果参考
                unique_df = pd.concat([s1_unique, s2_unique]).drop_duplicates(subset=['text'], keep='first').reset_index(drop=True)
                sentences = unique_df['text'].tolist()
                labels = unique_df['label'].tolist()
            
            with st.spinner(f"2/3 正在使用 GPU 将 {len(sentences)} 条语句编码为高维特征 (可能耗时数十秒)..."):
                np_embeddings = engine.encode(sentences, batch_size=256)
                norms = np.linalg.norm(np_embeddings, axis=1, keepdims=True)
                np_embeddings = np_embeddings / np.clip(norms, 1e-9, None)
                full_tensor = torch.tensor(np_embeddings).to(engine.device)
                
            with st.spinner("3/3 正在构建 FAISS 倒排索引..."):
                np_embeddings_f32 = np_embeddings.astype('float32')
                # 使用内积 (Inner Product) 索引评估，因为标准化后内积 == 余弦相似度
                faiss_index = faiss.IndexFlatIP(embed_dim) 
                faiss_index.add(np_embeddings_f32)

            # 保存到 session
            st.session_state.vector_db = {
                "tensor": full_tensor, 
                "faiss": faiss_index,             
                "texts": sentences,
                "labels": labels
            }
            st.success(f"成功构建向量库！一共编入 {len(sentences)} 根不重复向量。")

    if st.session_state.vector_db:
        st.divider()
        st.markdown("### 开始语义检索寻找相似语句")
        
        c_q1, c_q2 = st.columns([3, 1])
        with c_q1:
            query = st.text_input("请输入想要寻找相似句子的检索语句 (Query):", "北京天气咋样")
        with c_q2:
            top_k = st.slider("想要召回的数量 (Top-K):", min_value=1, max_value=50, value=10)
            
        if st.button("开始双轨平行检索", use_container_width=True):
            engine = get_loaded_engine()
            
            # 1. 编码用户的查询语句
            q_emb_np = engine.encode([query])
            norms = np.linalg.norm(q_emb_np, axis=1, keepdims=True)
            q_emb_np = q_emb_np / np.clip(norms, 1e-9, None)
            
            q_emb_tensor = torch.tensor(q_emb_np).to(engine.device)
            
            db = st.session_state.vector_db
                
            # 轨 1: PyTorch 纯矩阵暴力点乘算分
            pt_start = time.time()
            all_scores = torch.matmul(q_emb_tensor, db['tensor'].t()) 
            top_scores, top_indices = torch.topk(all_scores, k=top_k, dim=1)
            pt_end = time.time()
            pt_time_ms = (pt_end - pt_start) * 1000
            
            # 轨 2: FAISS 高速相似度查表算分
            np_q = q_emb_np.astype('float32')
            faiss_start = time.time()
            f_scores, f_indices = db['faiss'].search(np_q, top_k)
            faiss_end = time.time()
            faiss_time_ms = (faiss_end - faiss_start) * 1000
            st.markdown(f"**检索性能对比:** `PyTorch:` **{pt_time_ms:.2f} 毫秒** vs `FAISS:` **{faiss_time_ms:.2f} 毫秒**")
            
            # 组装召回结果为直观表格
            results = []
            
            idx_list = f_indices[0]
            score_list = f_scores[0]
                
            for r_i, (idx, score) in enumerate(zip(idx_list, score_list)):
                results.append({
                    "排名 (Rank)": r_i + 1,
                    "召回的句子 (Sentence)": db['texts'][idx],
                    "余弦相似度": f"{score:.4f}",
                    "原始语境判定(Label)": "由于环境关联 (1)" if db['labels'][idx] == 1 else "环境无关 (0)"
                })
                
            st.dataframe(pd.DataFrame(results), use_container_width=True)

