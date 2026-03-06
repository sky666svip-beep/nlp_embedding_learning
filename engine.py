import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
import sys
import pickle

import torch.nn as nn
from model import get_model
from data import get_dataloader, get_pretrained_dataloader
import pandas as pd
import numpy as np

class DualEncoderEngine:
    """
    统一的双塔模型训练器与推理引擎封装。
    完全对前端隔绝 PyTorch 的 device 管理、张量转换与 Batch 切分逻辑。
    新增任何模型，只需在 model.py 注册，前端即可一键复用全部功能。
    
    预训练模型 (pretrained) 走独立的分词器与数据加载分支：
    - 使用 HuggingFace AutoTokenizer 替代项目自建分词器
    - 训练时仅更新未冻结的参数
    """
    def __init__(self, model_type, embed_dim=128, vocab_size=None, model_state=None, tokenizer=None, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        # cuDNN 自动调优：为固定尺寸输入选择最快算法
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
        self.model_type = model_type
        self.embed_dim = embed_dim
        self.is_pretrained = (model_type in ("pretrained", "lora"))
        
        # 自动推断该架构需要的分词器类型
        if self.is_pretrained:
            self.tok_type = "pretrained"
        else:
            self.tok_type = "word" if model_type in ("cnn", "lstm", "transformer") else "char"
        
        self.tokenizer = tokenizer
        
        # 预训练模型需要 HuggingFace Tokenizer (本地优先，避免 403)
        if self.is_pretrained and self.tokenizer is None:
            from transformers import AutoTokenizer
            _name = "hfl/chinese-roberta-wwm-ext"
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(_name, local_files_only=True)
            except OSError:
                self.tokenizer = AutoTokenizer.from_pretrained(_name)
        
        # 初始化底座模型
        if self.is_pretrained:
            self.model = get_model(model_type, embed_dim=embed_dim).to(self.device)
            if model_state is not None:
                self.model.load_state_dict(model_state)
            self.model.eval()
        else:
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
        if self.is_pretrained:
            return self._train_pretrained(data_path, epochs, batch_size, lr, callback)
        return self._train_scratch(data_path, epochs, batch_size, lr, callback)
    
    def _train_scratch(self, data_path, epochs, batch_size, lr, callback):
        """手搭模型训练流程 (MeanPooling/CNN/LSTM/Transformer)"""
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
    
    def _train_pretrained(self, data_path, epochs, batch_size, lr, callback):
        """预训练模型微调流程 (RoBERTa) — 支持 AMP 混合精度加速"""
        dataloader = get_pretrained_dataloader(data_path, self.tokenizer, batch_size=batch_size)
        
        # 直接复用 __init__ 中已创建的模型，无需重复实例化
        self.model.train()
        
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
        criterion = nn.MSELoss()
        
        # AMP 混合精度：仅 CUDA 环境启用 (MPS/CPU 不支持)
        use_amp = (self.device.type == 'cuda')
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
        
        total_batches = len(dataloader)
        tag = "LoRA" if self.model_type == "lora" else "Pretrained"
        if use_amp:
            print(f"[{tag}] AMP 混合精度已启用 (FP16 加速)")
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            
            for batch_idx, (ids1, mask1, ids2, mask2, label) in enumerate(dataloader):
                ids1 = ids1.to(self.device, non_blocking=True)
                mask1 = mask1.to(self.device, non_blocking=True)
                ids2 = ids2.to(self.device, non_blocking=True)
                mask2 = mask2.to(self.device, non_blocking=True)
                label = label.to(self.device, non_blocking=True)
                
                optimizer.zero_grad(set_to_none=True)
                
                # AMP 自动混合精度前向传播
                with torch.amp.autocast('cuda', enabled=use_amp):
                    sim, _, _ = self.model(ids1, mask1, ids2, mask2)
                    target_sim = label * 2.0 - 1.0
                    loss = criterion(sim, target_sim)
                
                # AMP 缩放反向传播
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                
                correct = ((sim > 0) == (target_sim > 0)).float().sum()
                batch_acc = (correct / label.size(0)).item()
                
                total_loss += loss.item()
                
                if callback:
                    callback(epoch, batch_idx, loss.item(), batch_acc)
            
            avg_loss = total_loss / total_batches
            print(f"[{tag}] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        return self

    def encode(self, sentences, max_len=32, batch_size=256):
        """
        统一推理：大量句子转向量 (无需前端处理 Tensor)
        返回: numpy 矩阵 (N, embed_dim)
        """
        assert self.model is not None and self.tokenizer is not None, "Model or tokenizer not initialized for inference."
        self.model.eval()
        
        if self.is_pretrained:
            return self._encode_pretrained(sentences, max_len=128, batch_size=batch_size)
        return self._encode_scratch(sentences, max_len, batch_size)
    
    def _encode_scratch(self, sentences, max_len, batch_size):
        """手搭模型编码"""
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
    
    def _encode_pretrained(self, sentences, max_len, batch_size):
        """预训练模型编码"""
        all_vecs = []
        with torch.no_grad():
            for i in range(0, len(sentences), batch_size):
                batch_sentences = sentences[i:i+batch_size]
                encoded = self.tokenizer(
                    batch_sentences, max_length=max_len, padding='max_length',
                    truncation=True, return_tensors='pt'
                )
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)
                
                vec = self.model.encode_single(input_ids, attention_mask)
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
        
        if self.is_pretrained:
            return self._predict_pretrained(s1_list, s2_list, max_len=128, batch_size=batch_size)
        return self._predict_scratch(s1_list, s2_list, max_len, batch_size)
    
    def _predict_scratch(self, s1_list, s2_list, max_len, batch_size):
        """手搭模型预测"""
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
    
    def _predict_pretrained(self, s1_list, s2_list, max_len, batch_size):
        """预训练模型预测"""
        all_sims = []
        with torch.no_grad():
            for i in range(0, len(s1_list), batch_size):
                batch_s1 = s1_list[i:i+batch_size]
                batch_s2 = s2_list[i:i+batch_size]
                
                enc1 = self.tokenizer(batch_s1, max_length=max_len, padding='max_length',
                                      truncation=True, return_tensors='pt')
                enc2 = self.tokenizer(batch_s2, max_length=max_len, padding='max_length',
                                      truncation=True, return_tensors='pt')
                
                ids1 = enc1['input_ids'].to(self.device)
                mask1 = enc1['attention_mask'].to(self.device)
                ids2 = enc2['input_ids'].to(self.device)
                mask2 = enc2['attention_mask'].to(self.device)
                
                sim, _, _ = self.model(ids1, mask1, ids2, mask2)
                all_sims.append(sim.cpu().numpy())
        
        if len(all_sims) > 0:
            return np.concatenate(all_sims, axis=0)
        return np.array([])
