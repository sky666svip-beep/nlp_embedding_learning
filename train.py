import torch
import torch.nn as nn
from data import get_dataloader
from model import get_model

def train_model(data_path, epochs=10, batch_size=16, lr=1e-3, embed_dim=128, model_type="mean_pooling", callback=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    # CNN/LSTM 使用词级分词以捕获有意义的语义单元，MeanPooling 使用字符级
    tok_type = "word" if model_type in ("cnn", "lstm") else "char"
    print(f"Using device: {device} | Model: {model_type} | Tokenizer: {tok_type}")
    
    dataloader, tokenizer = get_dataloader(data_path, batch_size, tokenizer_type=tok_type)
    vocab_size = len(tokenizer.vocab)
    model = get_model(model_type, vocab_size, embed_dim).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for batch_idx, (id1, id2, label) in enumerate(dataloader):
            id1, id2, label = id1.to(device), id2.to(device), label.to(device)
            
            optimizer.zero_grad()
            sim, _, _ = model(id1, id2)
            
            # Map [0, 1] label to [-1, 1] to match cosine similarity output range
            target_sim = label * 2.0 - 1.0
            
            loss = criterion(sim, target_sim)
            loss.backward()
            # LSTM 梯度沿时间步反传容易爆炸，裁剪梯度范数防止高学习率下 NaN
            if model_type == "lstm":
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Calculate accuracy: positive sim for positive label, negative sim for negative label
            correct = ((sim > 0) == (target_sim > 0)).float().sum()
            batch_acc = (correct / label.size(0)).item()
            
            total_loss += loss.item()
            
            if callback:
                callback(epoch, batch_idx, loss.item(), batch_acc)
                
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
    return model, tokenizer

if __name__ == "__main__":
    # Test script for training
    model, tokenizer = train_model("data/lcqmc_2w.csv", epochs=2)
    print("Training test passed!")
