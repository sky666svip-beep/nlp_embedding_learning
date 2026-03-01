import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleDualEncoder(nn.Module):
    """极简双塔：Embedding + LayerNorm + MeanPooling + 投影层"""
    def __init__(self, vocab_size, embed_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=0)
        # 为适应 24w 数据的大词表稀疏性，添加 LayerNorm
        self.layer_norm = nn.LayerNorm(embed_dim)
        # 增加一个简单的线性映射，拉升词袋模型的表达上限
        self.projection = nn.Linear(embed_dim, embed_dim)
        
    def encode_single(self, seq):
        """将单个序列编码为句子向量"""
        mask = (seq != 0).float()
        emb = self.embedding(seq)
        emb = self.layer_norm(emb)
        vec = (emb * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
        vec = self.projection(vec)
        return vec

    def forward(self, seq1, seq2):
        vec1 = self.encode_single(seq1)
        vec2 = self.encode_single(seq2)
        sim = F.cosine_similarity(vec1, vec2, dim=-1)
        return sim, vec1, vec2


class CNNDualEncoder(nn.Module):
    """CNN 双塔：Embedding + LayerNorm + 多尺度 1D 卷积 + BatchNorm + 全局最大池化 + 投影层"""
    def __init__(self, vocab_size, embed_dim=128, kernel_sizes=(2, 3, 4), num_filters=64):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=0)
        # 为 CNN 增加 LayerNorm：因为 CNN 同样使用了 15773 的词级大词表，存在稀疏收敛问题
        self.layer_norm = nn.LayerNorm(embed_dim)
        # 多尺度卷积核：捕获词级 2-gram, 3-gram, 4-gram 局部特征
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=ks, padding=ks // 2),
                nn.BatchNorm1d(num_filters),
                nn.ReLU()
            )
            for ks in kernel_sizes
        ])
        self.dropout = nn.Dropout(0.3)
        # 投影层：将拼接后的卷积特征压缩回 embed_dim，使余弦相似度计算维度一致
        total_filters = num_filters * len(kernel_sizes)
        self.projection = nn.Linear(total_filters, embed_dim)

    def encode_single(self, seq):
        """将单个序列编码为句子向量"""
        emb = self.embedding(seq)                       # (batch, seq_len, embed_dim)
        emb = self.layer_norm(emb)                      # 稳定词向量输出
        emb = emb.permute(0, 2, 1)                      # (batch, embed_dim, seq_len) for Conv1d
        conv_outs = []
        for conv_block in self.convs:
            c = conv_block(emb)                         # (batch, num_filters, seq_len')
            pooled = c.max(dim=2).values                # (batch, num_filters) 全局最大池化
            conv_outs.append(pooled)
        vec = torch.cat(conv_outs, dim=1)               # (batch, num_filters * len(kernel_sizes))
        vec = self.dropout(vec)
        vec = self.projection(vec)                      # (batch, embed_dim) 压缩回统一维度
        return vec

    def forward(self, seq1, seq2):
        vec1 = self.encode_single(seq1)
        vec2 = self.encode_single(seq2)
        sim = F.cosine_similarity(vec1, vec2, dim=-1)
        return sim, vec1, vec2


class LSTMDualEncoder(nn.Module):
    """LSTM 双塔：Embedding + LayerNorm + 2层双向LSTM + Attention反向池化 + Dropout + 投影层"""
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=0)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            bidirectional=True, dropout=0.3 if num_layers > 1 else 0
        )
        # 注意力机制：用于计算序列中每个词语的重要性权重，替代简单粗暴的平均池化
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, 1)
        )
        self.dropout = nn.Dropout(0.3)
        self.projection = nn.Linear(hidden_dim * 2, embed_dim)

    def encode_single(self, seq):
        emb = self.embedding(seq)                            
        emb = self.layer_norm(emb)                           
        output, _ = self.lstm(emb)                           # (batch, seq_len, hidden_dim*2)
        
        # 核心增强：Attention Pooling (按词在句中的重要程度加权求和，而不是平均)
        attn_weights = self.attention(output).squeeze(-1)    # (batch, seq_len)
        attn_weights = attn_weights.masked_fill(seq == 0, -1e9)  # 遮蔽 PAD 标记
        attn_weights = F.softmax(attn_weights, dim=1)        # (batch, seq_len)
        
        vec = (output * attn_weights.unsqueeze(-1)).sum(dim=1) # (batch, hidden_dim*2)
        vec = self.dropout(vec)
        vec = self.projection(vec)                           # (batch, embed_dim)
        return vec

    def forward(self, seq1, seq2):
        vec1 = self.encode_single(seq1)
        vec2 = self.encode_single(seq2)
        sim = F.cosine_similarity(vec1, vec2, dim=-1)
        return sim, vec1, vec2


# 工厂函数：根据名称创建对应模型
MODEL_REGISTRY = {
    "mean_pooling": SimpleDualEncoder,
    "cnn": CNNDualEncoder,
    "lstm": LSTMDualEncoder,
}

def get_model(model_type, vocab_size, embed_dim=128):
    cls = MODEL_REGISTRY.get(model_type)
    if cls is None:
        raise ValueError(f"未知模型类型: {model_type}，可选: {list(MODEL_REGISTRY.keys())}")
    return cls(vocab_size, embed_dim)
