import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["DISABLE_SAFETENSORS_CONVERSION"] = "1"
import torch
import torch.nn as nn
import torch.nn.functional as F
def _load_roberta(model_name):
    """加载 RoBERTa 底座：本地缓存优先，无网络请求则不触发 403"""
    from transformers import AutoModel
    try:
        return AutoModel.from_pretrained(model_name, local_files_only=True)
    except OSError:
        print(f"[下载] 本地无缓存，正在从镜像站下载 {model_name}...")
        return AutoModel.from_pretrained(model_name)

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


class TransformerDualEncoder(nn.Module):
    """Transformer 双塔：Word Embed + Pos Embed + LayerNorm + 多头自注意力(TransformerEncoder) + MeanPooling + 投影层"""
    def __init__(self, vocab_size, embed_dim=128, num_heads=4, hidden_dim=256, num_layers=2, max_seq_len=64):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=0)
        # 绝对位置编码 (因为 Transformer 是排列不变的，必须告诉它词的位置)
        self.position_embedding = nn.Embedding(num_embeddings=max_seq_len, embedding_dim=embed_dim)
        
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # 定义单层 Transformer Encoder (Pre-LN 设置 norm_first=True 对从头训练极为关键)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim, 
            dropout=0.2, 
            batch_first=True,
            norm_first=True
        )
        # 堆叠多层
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 抛弃单纯的均值池化与生硬的线性映射 (极易导致表示坍塌到同一空间)
        # 我们引入像 LSTM 那样的大杀器：Self-Attention Pooling
        self.attention = nn.Linear(embed_dim, 1)

        # 核心：Transformer 从零训练极依赖合理的初始化
        self._init_weights()
        
    def _init_weights(self):
        # 词嵌入和位置嵌入通常使用标准差较小的正态分布
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0, std=0.02)
        # Attention层使用 Xavier
        nn.init.xavier_uniform_(self.attention.weight)
        if self.attention.bias is not None:
            nn.init.zeros_(self.attention.bias)

    def encode_single(self, seq):
        batch_size, seq_len = seq.size()
        
        # 1. 词嵌入与位置嵌入求和
        positions = torch.arange(seq_len, dtype=torch.long, device=seq.device).unsqueeze(0).expand(batch_size, seq_len)
        word_emb = self.embedding(seq)
        pos_emb = self.position_embedding(positions)
        emb = self.layer_norm(word_emb + pos_emb)
        
        # 2. 生成 pad mask。PyTorch 的 transformer 中，True 代表要被忽略的 Padding (和常规用法相反！)
        # seq shape: (batch, seq_len)
        padding_mask = (seq == 0) # (batch, seq_len), True 表示是 0
        
        # 3. 通过 Transformer
        # output: (batch, seq_len, embed_dim)
        output = self.transformer(emb, src_key_padding_mask=padding_mask)
        
        # 4. 使用 Attention Pooling 替代生硬的 Mean Pooling
        # output shape: (batch, seq_len, embed_dim)
        attn_weights = self.attention(output).squeeze(-1)    # (batch, seq_len)
        attn_weights = attn_weights.masked_fill(seq == 0, -1e9)  # 遮蔽 PAD 标记
        attn_weights = F.softmax(attn_weights, dim=1)        # (batch, seq_len)
        
        vec = (output * attn_weights.unsqueeze(-1)).sum(dim=1) # (batch, embed_dim)
        
        return vec

    def forward(self, seq1, seq2):
        vec1 = self.encode_single(seq1)
        vec2 = self.encode_single(seq2)
        sim = F.cosine_similarity(vec1, vec2, dim=-1)
        return sim, vec1, vec2


class PretrainedDualEncoder(nn.Module):
    """预训练双塔：RoBERTa-wwm-ext (冻结底层8层) + Attention Pooling + 投影层
    
    采用 B 方案微调策略：
    - 冻结: Embedding层 + Encoder Layer 0~7 (共8层)
    - 可训练: Encoder Layer 8~11 (共4层) + Attention Pooling + 投影层
    
    教学重点：让学习者直观感受"预训练底座提供的通用语义 vs 上层微调带来的任务适配"。
    """
    # 预训练模型名称 (哈工大全词掩码中文RoBERTa)
    PRETRAINED_NAME = "hfl/chinese-roberta-wwm-ext"
    # 冻结的 encoder 层数 (底部8层不参与训练)
    FREEZE_LAYERS = 8

    def __init__(self, embed_dim=128):
        super().__init__()
        self.roberta = _load_roberta(self.PRETRAINED_NAME)
        hidden_size = self.roberta.config.hidden_size  # 768
        
        # 冻结策略：固定 Embedding + 底部 8 层 Encoder
        self._freeze_base_layers()
        
        # 自注意力池化：复用项目中已验证有效的 Attention Pooling 机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # 投影层：将 RoBERTa 的 768 维压缩到项目统一的 embed_dim
        self.projection = nn.Linear(hidden_size, embed_dim)
        self.dropout = nn.Dropout(0.1)
    
    def _freeze_base_layers(self):
        """冻结 Embedding 层和底部 FREEZE_LAYERS 层 Encoder"""
        # 冻结词嵌入层
        for param in self.roberta.embeddings.parameters():
            param.requires_grad = False
        
        # 冻结底部的 encoder 层
        for i in range(self.FREEZE_LAYERS):
            for param in self.roberta.encoder.layer[i].parameters():
                param.requires_grad = False
    
    def encode_single(self, input_ids, attention_mask):
        """将单个序列编码为句子向量
        
        与其他手搭模型不同，预训练模型需要 attention_mask 来区分有效 token 和 padding。
        """
        # RoBERTa 前向传播：获取所有层的隐状态
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, 768)
        
        # Attention Pooling：计算每个 token 的重要性权重
        attn_weights = self.attention(hidden_states).squeeze(-1)  # (batch, seq_len)
        # 用 attention_mask 遮蔽 padding 位置 (mask=0 的位置设为极小值)
        attn_weights = attn_weights.masked_fill(attention_mask == 0, -1e9)
        attn_weights = F.softmax(attn_weights, dim=1)  # (batch, seq_len)
        
        # 加权求和得到句子级向量
        vec = (hidden_states * attn_weights.unsqueeze(-1)).sum(dim=1)  # (batch, 768)
        vec = self.dropout(vec)
        vec = self.projection(vec)  # (batch, embed_dim)
        return vec
    
    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        """双塔前向传播：两段输入分别编码后计算余弦相似度"""
        vec1 = self.encode_single(input_ids_1, attention_mask_1)
        vec2 = self.encode_single(input_ids_2, attention_mask_2)
        sim = F.cosine_similarity(vec1, vec2, dim=-1)
        return sim, vec1, vec2


class LoRADualEncoder(nn.Module):
    """LoRA 双塔：RoBERTa-wwm-ext + LoRA 低秩适应 + Attention Pooling + 投影层
    
    全量冻结 RoBERTa 所有原始参数，仅通过 LoRA 在 query/value 矩阵中
    注入低秩可训练参数。可训练参数量约 30~50 万 (对比冻结层方案的 2963 万)。
    
    教学重点：
    - LoRA 不修改原始权重，而是给每个目标矩阵旁路注入 A*B 低秩分解
    - r=8 意味着每个注入点只增加 768*8 + 8*768 = 12288 个参数
    - 训练效率高：显存占用比冻结层方案低 10%-20%
    """
    # 复用预训练模型名称
    PRETRAINED_NAME = "hfl/chinese-roberta-wwm-ext"
    # LoRA 超参数 (经典配置)
    LORA_R = 8           # 低秩维度
    LORA_ALPHA = 16      # 缩放因子 (alpha/r = 2 倍缩放)
    LORA_DROPOUT = 0.1   # LoRA 层 Dropout
    # 注入目标：自注意力的 Query 和 Value 矩阵
    LORA_TARGET_MODULES = ["query", "value"]
    
    def __init__(self, embed_dim=128):
        super().__init__()
        # 第一步：加载原始 RoBERTa (hidden_size=768, 12层 Transformer)
        from peft import get_peft_model, LoraConfig, TaskType
        base_model = _load_roberta(self.PRETRAINED_NAME)
        hidden_size = base_model.config.hidden_size  # 768
        
        # 第二步：用 LoRA 包裹 —— 自动冻结所有原始参数，只有 LoRA 矩阵可训练
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=self.LORA_R,
            lora_alpha=self.LORA_ALPHA,
            lora_dropout=self.LORA_DROPOUT,
            target_modules=self.LORA_TARGET_MODULES,
        )
        self.roberta = get_peft_model(base_model, lora_config)
        
        # 打印可训练参数统计 (教学用)
        self.roberta.print_trainable_parameters()
        
        # 第三步：自注意力池化 + 投影层 (与冻结层方案共享相同结构)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        self.projection = nn.Linear(hidden_size, embed_dim)
        self.dropout = nn.Dropout(0.1)
    
    def encode_single(self, input_ids, attention_mask):
        """将单个序列编码为句子向量 (接口与 PretrainedDualEncoder 完全一致)"""
        # LoRA 包裹后的 RoBERTa 前向传播：低秩旁路自动生效
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, 768)
        
        # Attention Pooling
        attn_weights = self.attention(hidden_states).squeeze(-1)
        attn_weights = attn_weights.masked_fill(attention_mask == 0, -1e9)
        attn_weights = F.softmax(attn_weights, dim=1)
        
        vec = (hidden_states * attn_weights.unsqueeze(-1)).sum(dim=1)
        vec = self.dropout(vec)
        vec = self.projection(vec)
        return vec
    
    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        """双塔前向传播：两段输入分别编码后计算余弦相似度"""
        vec1 = self.encode_single(input_ids_1, attention_mask_1)
        vec2 = self.encode_single(input_ids_2, attention_mask_2)
        sim = F.cosine_similarity(vec1, vec2, dim=-1)
        return sim, vec1, vec2


# 工厂函数：根据名称创建对应模型
MODEL_REGISTRY = {
    "mean_pooling": SimpleDualEncoder,
    "cnn": CNNDualEncoder,
    "lstm": LSTMDualEncoder,
    "transformer": TransformerDualEncoder,
    "pretrained": PretrainedDualEncoder,
    "lora": LoRADualEncoder,
}

def get_model(model_type, vocab_size=0, embed_dim=128):
    """根据模型类型创建对应的双塔模型实例。
    
    预训练模型不需要 vocab_size (自带词表)，其他手搭模型需要。
    """
    cls = MODEL_REGISTRY.get(model_type)
    if cls is None:
        raise ValueError(f"未知模型类型: {model_type}，可选: {list(MODEL_REGISTRY.keys())}")
    
    # 预训练模型 (冻结层/LoRA) 只需要 embed_dim，不需要 vocab_size
    if model_type in ("pretrained", "lora"):
        return cls(embed_dim=embed_dim)
    return cls(vocab_size, embed_dim)
