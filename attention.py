import torch
from torch import nn
from torch.nn import Parameter, LayerNorm

class Attention(nn.Module):
    def __init__(self, embed_dim, ctx_len, mem_len):
        super().__init__()
        self.w_q = nn.Linear(embed_dim, embed_dim)
        self.w_ke = nn.Linear(embed_dim, embed_dim)
        self.w_kr = nn.Linear(embed_dim, embed_dim)
        self.w_v = nn.Linear(embed_dim, embed_dim)
        self.mlp = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = LayerNorm(embed_dim)
        
        self.pos_ff = nn.Sequential(
            nn.Linear(embed_dim, 4*embed_dim),
            nn.ReLU(),
            nn.Linear(4*embed_dim, embed_dim)
        )
        
        self.mem = torch.zeros(mem_len, embed_dim, requires_grad=False)
        
        self.u = Parameter(torch.randn(ctx_len, embed_dim))
        self.v = Parameter(torch.randn(ctx_len, embed_dim))
        
        r = self.get_sinusoid_pos_encoding(ctx_len+mem_len, embed_dim)
        self.r = torch.flip(r, dims=(0,))
    
    def forward(self, x, h):
        h = torch.cat((h, x), dim=1)
        q = self.w_q(x)
        k = self.w_ke(h)
        v = self.w_v(h)
        
        _, ctx_len, embed_dim = x.shape
        mem_len = h.shape[1] - ctx_len
        
        b = q @ self.w_kr(self.r).T
        b = self.circulant_shift(b, -ctx_len+1) # do we need to tril here or is att.tril fine?
                
        att = q @ k.T + b + self.u @ k.T + self.v @ self.w_kr(self.r).T
        att = att.tril(mem_len) / embed_dim**0.5
        att = torch.softmax(att, dim=-1)
        
        out = self.layer_norm(self.mlp(att @ v))
        return self.pos_ff(out)
          
    def get_sinusoid_pos_encoding(self, mem_len, embed_dim):
        pos = torch.arange(mem_len).unsqueeze(1)
        enc = torch.arange(embed_dim).float()
        enc = enc.unsqueeze(0).repeat(mem_len, 1)
        enc[:, ::2] = torch.sin(pos / 10000**(2*enc[:, ::2]/embed_dim))
        enc[:, 1::2] = torch.cos(pos / 10000**(2*enc[:, 1::2]/embed_dim))
        return enc
    
    def circulant_shift(self, x, shift):
        """
        Shifts top row of `x` by `shift`, second row by `shift-1`, etc. This is
        used when computing the relative positional encoding matrix in linear
        time (as opposed to quadratic time for the naive solution).
        
        See Appendix B of the Transformer-XL paper for more details.
        """
        batch_size, height, width = x.shape
        i = torch.arange(width).roll(shift).unsqueeze(0)
        i = i.flip(1).repeat(1, 2)
        i = i.unfold(dimension=1, size=width, step=1).flip(-1)
        i = i.repeat(batch_size, 1, 1)[:, :height]
        return x.gather(2, i)

batch_size = 64
embed_dim = 512
ctx_len = 100
x = torch.randn(batch_size, ctx_len, embed_dim)
h = torch.randn(batch_size, 384, embed_dim)
att = Attention(embed_dim, ctx_len, mem_len=384)
out = att(x, h)