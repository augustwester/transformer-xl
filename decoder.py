from torch import nn
from attention import MultiHeadAttention

class PositionwiseFeedForward(nn.Module):
    def __init__(self, model_dim, inner_dim):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(model_dim, inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim, model_dim)
        )
        
    def forward(self, x):
        return self.net(x)

class DecoderLayer(nn.Module):
    def __init__(self, model_dim, embed_dim, mem_len, num_heads, inner_dim, R, device):
        super().__init__()
        self.attn = MultiHeadAttention(model_dim, embed_dim, mem_len, num_heads, R, device)
        self.pos_ff = PositionwiseFeedForward(model_dim, inner_dim)
    
    def forward(self, x, mem, att_mask):
        att_out = self.attn(x, mem, att_mask)
        return self.pos_ff(att_out)