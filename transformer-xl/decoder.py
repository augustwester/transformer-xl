from torch import nn
from attention import MultiHeadAttention

class PositionwiseFeedForward(nn.Module):
    def __init__(self, model_dim, inner_dim, dropout):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(model_dim, inner_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(inner_dim, model_dim),
            nn.Dropout(dropout)
        )
        self.layer_norm = nn.LayerNorm(model_dim)
        
    def forward(self, x):
        return self.layer_norm(self.net(x) + x)

class DecoderLayer(nn.Module):
    def __init__(self, model_dim, embed_dim, mem_len, num_heads, inner_dim, dropout, R, device):
        super().__init__()
        self.attn = MultiHeadAttention(model_dim, embed_dim, mem_len, num_heads, dropout, R, device)
        self.pos_ff = PositionwiseFeedForward(model_dim, inner_dim, dropout)
    
    def forward(self, x, mem):
        att_out = self.attn(x, mem)
        return self.pos_ff(att_out)