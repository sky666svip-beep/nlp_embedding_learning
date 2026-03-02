import torch
import torch.nn as nn
from model import get_model
from data import get_dataloader
import pandas as pd
import numpy as np

class DualEncoderEngine:
    """
    统一的双塔模型训练器与推理引擎封装。
    完全对前端隔绝 PyTorch 的 device 管理、张量转换与 Batch 切分逻辑。
    新增任何模型，只需在 model.py 注册，前端即可一键复用全部功能。
    """
    def __init__(self, model_type, embed_dim=128, vocab_size=None, model_state=None, tokenizer=None, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        self.model_type = model_type
        self.embed_dim = embed_dim
        
        # 自动推断该架构需要的分词器类型
        self.tok_type = "word" if model_type in ("cnn", "lstm", "transformer") else "char"
        
        self.tokenizer = tokenizer
        
        # 如果提供了预训练环境或准备好推理，则初始化底座模型
        if vocab_size is None and self.tokenizer is not None:
            vocab_size = len(self.tokenizer.vocab)
            
        if vocab_size is not None:
            self.model = get_model(model_type, vocab_size, embed_dim).to(self.device)
            if model_state is not None:
                self.model.load_state_dict(model_state)
            self.model.eval()
        else:
            self.model = None

    def train(self, data_path, epochs=10, batch_size=16, lr=1e-3, callback=None):
        """统一训练接口"""
        dataloader, self.tokenizer = get_dataloader(data_path, batch_size, tokenizer_type=self.tok_type)
        vocab_size = len(self.tokenizer.vocab)
        self.model = get_model(self.model_type, vocab_size, self.embed_dim).to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            
            for batch_idx, (id1, id2, label) in enumerate(dataloader):
                id1, id2, label = id1.to(self.device), id2.to(self.device), label.to(self.device)
                
                optimizer.zero_grad()
                sim, _, _ = self.model(id1, id2)
                
                # 标签域从 [0, 1] 映射到 [-1, 1]
                target_sim = label * 2.0 - 1.0
                
                loss = criterion(sim, target_sim)
                loss.backward()
                
                # LSTM 和 Transformer 的梯度裁剪保护，防止梯度爆炸
                if self.model_type in ("lstm", "transformer"):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                correct = ((sim > 0) == (target_sim > 0)).float().sum()
                batch_acc = (correct / label.size(0)).item()
                
                total_loss += loss.item()
                
                if callback:
                    callback(epoch, batch_idx, loss.item(), batch_acc)
                    
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            
        return self

    def encode(self, sentences, max_len=32, batch_size=256):
        """
        统一推理：大量句子转向量 (无需前端处理 Tensor)
        返回: numpy 矩阵 (N, embed_dim)
        """
        assert self.model is not None and self.tokenizer is not None, "Model or tokenizer not initialized for inference."
        self.model.eval()
        all_vecs = []
        with torch.no_grad():
            for i in range(0, len(sentences), batch_size):
                batch_sentences = sentences[i:i+batch_size]
                ids = [torch.tensor(self.tokenizer.encode(s, max_len=max_len)) for s in batch_sentences]
                input_tensor = torch.stack(ids).to(self.device)
                
                vec = self.model.encode_single(input_tensor)
                all_vecs.append(vec.cpu().numpy())
                
        if len(all_vecs) > 0:
            return np.concatenate(all_vecs, axis=0)
        return np.array([])

    def predict_similarity(self, s1_list, s2_list, max_len=32, batch_size=256):
        """
        统一推理：批处理计算两组句子的相似度
        返回: (N,) numpy 数组，包含每对句子的余弦打分
        """
        assert self.model is not None and self.tokenizer is not None, "Model or tokenizer not initialized for inference."
        assert len(s1_list) == len(s2_list), "s1_list and s2_list must have the same length"
        
        self.model.eval()
        all_sims = []
        with torch.no_grad():
            for i in range(0, len(s1_list), batch_size):
                batch_s1 = s1_list[i:i+batch_size]
                batch_s2 = s2_list[i:i+batch_size]
                
                id1 = torch.stack([torch.tensor(self.tokenizer.encode(s, max_len=max_len)) for s in batch_s1]).to(self.device)
                id2 = torch.stack([torch.tensor(self.tokenizer.encode(s, max_len=max_len)) for s in batch_s2]).to(self.device)
                
                sim, _, _ = self.model(id1, id2)
                all_sims.append(sim.cpu().numpy())
                
        if len(all_sims) > 0:
            return np.concatenate(all_sims, axis=0)
        return np.array([])
