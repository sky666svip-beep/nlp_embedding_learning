import hashlib
import os
import pickle
import re
import jieba
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

def clean_text(text):
    """轻量级文本清洗：去除 HTML 标签、标点符号和特殊字符，只保留中文、字母和数字"""
    text = str(text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^\u4e00-\u9fffA-Za-z0-9]', '', text)
    return text

class SimpleCharTokenizer:
    """字符级分词器：逐字拆分，适合 MeanPooling 模型"""
    def __init__(self):
        self.vocab = {"[PAD]": 0, "[UNK]": 1}
        self.id2char = {0: "[PAD]", 1: "[UNK]"}
        
    def fit(self, texts):
        for text in texts:
            text = clean_text(text)
            for char in text:
                if char not in self.vocab:
                    idx = len(self.vocab)
                    self.vocab[char] = idx
                    self.id2char[idx] = char
                    
    def encode(self, text, max_len=32):
        text = clean_text(text)
        ids = [self.vocab.get(char, 1) for char in text]
        if len(ids) > max_len:
            ids = ids[:max_len]
        else:
            ids = ids + [0] * (max_len - len(ids))
        return ids

class SimpleWordTokenizer:
    """词级分词器：使用 jieba 分词，适合 CNN 模型捕获词级 N-gram"""
    def __init__(self):
        self.vocab = {"[PAD]": 0, "[UNK]": 1}
        self.id2word = {0: "[PAD]", 1: "[UNK]"}
        
    def fit(self, texts):
        for text in texts:
            text = clean_text(text)
            for word in jieba.cut(text):
                if word not in self.vocab:
                    idx = len(self.vocab)
                    self.vocab[word] = idx
                    self.id2word[idx] = word
                    
    def encode(self, text, max_len=20):
        text = clean_text(text)
        words = list(jieba.cut(text))
        ids = [self.vocab.get(w, 1) for w in words]
        if len(ids) > max_len:
            ids = ids[:max_len]
        else:
            ids = ids + [0] * (max_len - len(ids))
        return ids

class STSDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=32):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = [(clean_text(row['sentence1']), clean_text(row['sentence2']), int(row['label']))
                     for _, row in df.iterrows()]
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        s1, s2, label = self.data[idx]
        id1 = torch.tensor(self.tokenizer.encode(s1, self.max_len), dtype=torch.long)
        id2 = torch.tensor(self.tokenizer.encode(s2, self.max_len), dtype=torch.long)
        label = torch.tensor(label, dtype=torch.float32)
        return id1, id2, label

def get_dataloader(csv_path, batch_size=16, tokenizer=None, max_len=32, tokenizer_type="char"):
    """tokenizer_type: 'char' (字符级) 或 'word' (词级，jieba)
    加入本地离线缓存机制，实现大文件的秒级序列化加载。
    """
    # 建立缓存标识
    cache_dir = "data/cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    file_stat = os.stat(csv_path)
    # 基于路径、修改时间、分词类型组合哈希
    hash_str = f"{csv_path}_{file_stat.st_mtime}_{tokenizer_type}_{max_len}"
    cache_key = hashlib.md5(hash_str.encode()).hexdigest()
    
    dataset_cache_path = os.path.join(cache_dir, f"dataset_{cache_key}.pkl")
    tokenizer_cache_path = os.path.join(cache_dir, f"tokenizer_{cache_key}.pkl")
    
    if not tokenizer:
        if os.path.exists(dataset_cache_path) and os.path.exists(tokenizer_cache_path):
            print(f"👉 命中本地缓存: {cache_key}，正在极速加载序列化数据集...")
            with open(tokenizer_cache_path, "rb") as f:
                tokenizer = pickle.load(f)
            with open(dataset_cache_path, "rb") as f:
                dataset = pickle.load(f)
            return DataLoader(dataset, batch_size=batch_size, shuffle=True), tokenizer

        print(f"⏳ 未命中本地缓存: 正在全量分词与张量化 {csv_path} (请耐心等待)...")
        df = pd.read_csv(csv_path)
        
        if tokenizer_type == "word":
            tokenizer = SimpleWordTokenizer()
            max_len = 20  # 词级分词后序列更短
        else:
            tokenizer = SimpleCharTokenizer()
        
        # 提取语料训练分词器
        texts = df['sentence1'].tolist() + df['sentence2'].tolist()
        tokenizer.fit(texts)
        
        # 实例化 Dataset（会在内部进行 encode）
        dataset = STSDataset(df, tokenizer, max_len)
        
        # 落盘缓存
        with open(tokenizer_cache_path, "wb") as f:
            pickle.dump(tokenizer, f)
        with open(dataset_cache_path, "wb") as f:
            pickle.dump(dataset, f)
            
        print(f"✅ 生成静态缓存完成: {cache_key}")
    else:
        # 如果从外部传入了已训练好的 tokenizer（例如预测时），则直连不缓存
        df = pd.read_csv(csv_path)
        dataset = STSDataset(df, tokenizer, max_len)

    return DataLoader(dataset, batch_size=batch_size, shuffle=True), tokenizer
