import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from train import train_model
from model import SimpleDualEncoder
from data import get_dataloader
import torch
import os
import pickle

st.set_page_config(page_title="NLP Embedding 学习", layout="wide")
st.title("NLP Embedding 极简双塔模型")

# Sidebar
st.sidebar.header("⚙️ 训练参数")
epochs = st.sidebar.slider("Epochs", min_value=1, max_value=50, value=15)
lr = st.sidebar.number_input("Learning Rate", value=0.005, format="%.4f")
batch_size = st.sidebar.selectbox("Batch Size", [4, 8, 16, 32], index=2)
embed_dim = st.sidebar.slider("词向量维度", min_value=8, max_value=256, value=64)

if "model_state" not in st.session_state:
    st.session_state.model_state = None
if "tokenizer" not in st.session_state:
    st.session_state.tokenizer = None
if "loss_history" not in st.session_state:
    st.session_state.loss_history = []

tab1, tab2, tab3 = st.tabs(["📊 1. 训练监控", "🌌 2. 空间特征降维", "🤖 3. 模型预测"])

with st.sidebar:
    start_train = st.button("🚀 开始训练", use_container_width=True)
    
    st.divider()
    if st.session_state.model_state is not None:
        if st.button("💾 保存模型到本地", use_container_width=True):
            os.makedirs("output", exist_ok=True)
            # Save weights
            torch.save(st.session_state.model_state, "output/model_weights.pth")
            # Save tokenizer vocabulary simply using pickle
            with open("output/tokenizer.pkl", "wb") as f:
                pickle.dump(st.session_state.tokenizer, f)
            st.success("模型权重与词表已保存至 `output/` 目录！")

if start_train:
    st.session_state.loss_history = []
    
    with tab1:
        st.subheader("Training Loss 实时下降曲线")
        progress_bar = st.progress(0)
        status_text = st.empty()
        chart_element = st.empty()
        
        # Determine total batches
        mock_dl, mock_tk = get_dataloader("data/lcqmc_mini.csv", batch_size=batch_size)
        total_batches = len(mock_dl)
        total_steps = epochs * total_batches
        
        def train_callback(epoch, batch_idx, loss):
            st.session_state.loss_history.append(loss)
            current_step = epoch * total_batches + batch_idx + 1
            progress_bar.progress(current_step / total_steps)
            status_text.text(f"进程: Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{total_batches} | 实时Loss: {loss:.4f}")
            chart_element.line_chart(st.session_state.loss_history)
            
        with st.spinner("模型大脑正在学习语义分布中..."):
            model, tokenizer = train_model("data/lcqmc_mini.csv", epochs, batch_size, lr, embed_dim, callback=train_callback)
            st.session_state.model_state = model.state_dict()
            st.session_state.tokenizer = tokenizer
            
        st.success("🎉 训练完成！请前往其他 Tab 查看词频空间降维效果和句子相似度。")
else:
    with tab1:
        st.subheader("Training Loss 实时下降曲线")
        if st.session_state.loss_history:
            st.line_chart(st.session_state.loss_history)
            st.success("🎉 这是当前存留的模型训练走势。")
        else:
            st.info("👈 请先在侧边栏点击【开始训练】。")

def get_loaded_model():
    if not st.session_state.model_state or not st.session_state.tokenizer:
        return None, None
    tk = st.session_state.tokenizer
    mod = SimpleDualEncoder(len(tk.vocab), embed_dim)
    mod.load_state_dict(st.session_state.model_state)
    mod.eval()
    return mod, tk

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

with tab2:
    st.subheader("二维句子空间分布 (PCA降维)")
    st.markdown("将高维的句子向量压缩至2D平面，距离相近的点代表模型认为它们语义相似。")
    if st.session_state.model_state:
        mod, tk = get_loaded_model()
        df = pd.read_csv("data/lcqmc_mini.csv")
        sentences = list(set(df['sentence1'].tolist()[:30] + df['sentence2'].tolist()[:30]))
        
        with st.spinner("抽取高维特征并计算 PCA..."):
            mod = mod.to(device)
            embs = []
            for s in sentences:
                ids = torch.tensor([tk.encode(s, max_len=32)], dtype=torch.long).to(device)
                mask = (ids != 0).float()
                emb = mod.embedding(ids)
                vec = (emb * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
                embs.append(vec.squeeze(0).cpu().detach().numpy())
            
            coords = PCA(n_components=2).fit_transform(np.array(embs))
            chart_df = pd.DataFrame({"X": coords[:, 0], "Y": coords[:, 1], "Text": sentences})
            
            # Use columns to lay out side-by-side
            c1, c2 = st.columns([2, 1])
            with c1:
                st.scatter_chart(chart_df, x="X", y="Y")
            with c2:
                st.dataframe(chart_df[["X", "Y", "Text"]], use_container_width=True)
                st.caption("👈 左侧图表为点位分布，对照此表可查找具体句子所在坐标。")
    else:
        st.info("👈 请先在侧边栏点击【开始训练】。")

with tab3:
    st.subheader("输入两句话，预测相似度指数")
    mod, tk = get_loaded_model()
    
    col1, col2 = st.columns(2)
    s1 = col1.text_input("第一句话", "苹果手机怎么截图")
    s2 = col2.text_input("第二句话", "iPhone屏显怎么截")
        
    if st.button("⚡ 计算相似度", type="primary"):
        if mod:
            mod = mod.to(device)
            id1 = torch.tensor([tk.encode(s1, max_len=32)], dtype=torch.long).to(device)
            id2 = torch.tensor([tk.encode(s2, max_len=32)], dtype=torch.long).to(device)
            sim, _, _ = mod(id1, id2)
            similarity_score = sim.cpu().item()
            
            st.metric(label="Cosine Similarity (余弦相似度)", value=f"{similarity_score:.4f}")
            if similarity_score > 0.5:
                st.success("💡 判断：它们大概率是 **相似** 的！")
            else:
                st.warning("⏳ 判断：它们可能 **不相似**。")
        else:
            st.error("⚠️ 模型未初始化，请先训练！")
