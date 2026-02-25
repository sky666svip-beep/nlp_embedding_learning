import torch
import torch.nn as nn
from data import get_dataloader
from model import SimpleDualEncoder

def train_model(data_path, epochs=10, batch_size=16, lr=1e-3, embed_dim=128, callback=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    dataloader, tokenizer = get_dataloader(data_path, batch_size)
    vocab_size = len(tokenizer.vocab)
    model = SimpleDualEncoder(vocab_size, embed_dim).to(device)
    
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
            optimizer.step()
            
            total_loss += loss.item()
            
            if callback:
                callback(epoch, batch_idx, loss.item())
                
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
    return model, tokenizer

if __name__ == "__main__":
    # Test script for training
    model, tokenizer = train_model("data/lcqmc_mini.csv", epochs=2)
    print("Training test passed!")
