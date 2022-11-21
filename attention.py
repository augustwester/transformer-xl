import torch
from torch import nn
from torch.nn import Parameter, LayerNorm

class Attention(nn.Module):
    def __init__(self, embed_dim, seg_len, mem_len, num_heads):
        super().__init__()
        
        # should these have identical rows???
        self.u = Parameter(torch.randn(num_heads, seg_len, embed_dim))
        self.v = Parameter(torch.randn(num_heads, seg_len, embed_dim))
        
        self.w_q = nn.Linear(embed_dim, num_heads*embed_dim)
        self.w_ke = nn.Linear(embed_dim, num_heads*embed_dim)
        self.w_kr = nn.Linear(embed_dim, num_heads*embed_dim)
        self.w_v = nn.Linear(embed_dim, num_heads*embed_dim)
        self.mlp = nn.Linear(num_heads*embed_dim, embed_dim)
        self.layer_norm = LayerNorm(embed_dim)
        
        self.pos_ff = nn.Sequential(
            nn.Linear(embed_dim, 4*embed_dim),
            nn.ReLU(),
            nn.Linear(4*embed_dim, embed_dim)
        )
        
        self.mem_len = mem_len
        self.mem = torch.empty(0, 0, embed_dim, requires_grad=False)
        
        pos = self.get_sinusoid_pos_encoding(seg_len+mem_len, embed_dim)
        self.pos = torch.flip(pos, dims=(0,))
    
    def forward(self, x, mem):
        # concat output from previous layer with "memory" from earlier segments
        h = torch.cat((mem, x), dim=1)
        
        batch_size, seg_len, embed_dim = x.shape
        mem_len = h.shape[1] - seg_len
        total_len = h.shape[1]
        
        # compute projections of output from previous layer and the memory
        q = self.w_q(x).reshape(batch_size, -1, seg_len, embed_dim)
        k = self.w_ke(h).reshape(batch_size, -1, total_len, embed_dim)
        v = self.w_v(h).reshape(batch_size, -1, total_len, embed_dim)
        r = self.w_kr(self.pos).reshape(-1, total_len, embed_dim)
        
        # compute relative positional encodings
        b = q @ r.transpose(1, 2)
        b = self.circulant_shift(b, -seg_len+1)
        
        # this is the XL specific way of computing the attention score
        k = k.transpose(2, 3)
        att = q @ k + b + self.u @ k + self.v @ r.transpose(1, 2)
        att = att.tril(mem_len) / embed_dim**0.5
        att = torch.softmax(att, dim=-1)
        
        # compute the output of the layer and save to memory
        out = self.layer_norm(self.mlp(att @ v) + x)
        out = self.pos_ff(out)
        self.save_to_memory(out)
        
        return out
          
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
        used to compute the relative positional encoding matrix in linear time
        (as opposed to quadratic time for the naive solution).
        
        See Appendix B of the Transformer-XL paper for more details.
        """
        batch_size, num_heads, height, width = x.shape
        i = torch.arange(width).roll(shift).unsqueeze(0)
        i = i.flip(1).repeat(1, 2)
        i = i.unfold(dimension=1, size=width, step=1).flip(-1).unsqueeze(0)
        i = i.repeat(batch_size, num_heads, 1, 1)[:, :, :height]
        return x.gather(3, i)
    
    def save_to_memory(self, h):
        self.mem = torch.stack((self.mem, h), dim=1)[:, :self.mem_len]

batch_size = 64
embed_dim = 512
seg_len = 100
x = torch.randn(batch_size, seg_len, embed_dim)
h = torch.randn(batch_size, 384, embed_dim)
att = Attention(embed_dim, seg_len, mem_len=384, num_heads=10)
out = att(x, h)