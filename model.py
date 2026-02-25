import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleDualEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=0)
        
    def forward(self, seq1, seq2):
        mask1 = (seq1 != 0).float()
        mask2 = (seq2 != 0).float()
        
        emb1 = self.embedding(seq1)
        emb2 = self.embedding(seq2)
        
        vec1 = (emb1 * mask1.unsqueeze(-1)).sum(dim=1) / mask1.sum(dim=1, keepdim=True).clamp(min=1e-9)
        vec2 = (emb2 * mask2.unsqueeze(-1)).sum(dim=1) / mask2.sum(dim=1, keepdim=True).clamp(min=1e-9)
        
        sim = F.cosine_similarity(vec1, vec2, dim=-1)
        return sim, vec1, vec2
