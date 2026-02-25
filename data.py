import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class SimpleCharTokenizer:
    def __init__(self):
        self.vocab = {"[PAD]": 0, "[UNK]": 1}
        self.id2char = {0: "[PAD]", 1: "[UNK]"}
        
    def fit(self, texts):
        for text in texts:
            for char in text:
                if char not in self.vocab:
                    idx = len(self.vocab)
                    self.vocab[char] = idx
                    self.id2char[idx] = char
                    
    def encode(self, text, max_len=32):
        ids = [self.vocab.get(char, 1) for char in text]
        if len(ids) > max_len:
            ids = ids[:max_len]
        else:
            ids = ids + [0] * (max_len - len(ids))
        return ids

class STSDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=32):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = [(row['sentence1'], row['sentence2'], int(row['label'])) for _, row in df.iterrows()]
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        s1, s2, label = self.data[idx]
        id1 = torch.tensor(self.tokenizer.encode(s1, self.max_len), dtype=torch.long)
        id2 = torch.tensor(self.tokenizer.encode(s2, self.max_len), dtype=torch.long)
        label = torch.tensor(label, dtype=torch.float32)
        return id1, id2, label

def get_dataloader(csv_path, batch_size=16, tokenizer=None, max_len=32):
    df = pd.read_csv(csv_path)
    if not tokenizer:
        tokenizer = SimpleCharTokenizer()
        tokenizer.fit(df['sentence1'].tolist() + df['sentence2'].tolist())
    dataset = STSDataset(df, tokenizer, max_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True), tokenizer
