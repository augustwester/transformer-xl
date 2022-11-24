import torch
from torch import nn
        
class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, embed_dim, seg_len, mem_len, num_heads, device):
        super().__init__()
        
        self.mem_len = mem_len
        self.embed_dim = embed_dim
        
        self.u1 = nn.Parameter(torch.randn(num_heads, 1, embed_dim))
        self.u2 = nn.Parameter(torch.randn(num_heads, 1, embed_dim))
        
        self.w_q = nn.Linear(model_dim, num_heads*embed_dim, bias=False)
        self.w_kv = nn.Linear(model_dim, 2*num_heads*embed_dim, bias=False)
        self.w_r = nn.Linear(model_dim, num_heads*embed_dim, bias=False)
        self.mlp = nn.Linear(num_heads*embed_dim, model_dim, bias=False)
        self.layer_norm = nn.LayerNorm(model_dim)
        
        pos = self.get_sinusoid_pos_encoding(seg_len+mem_len, embed_dim).to(device)
        self.pos = torch.flip(pos, dims=(0,))
    
    def forward(self, x, mem, att_mask):
        # concat output from previous layer with "memory" from earlier segments
        h = torch.cat((mem, x), dim=1)
        
        batch_size, seg_len, embed_dim = x.shape
        mem_len = h.shape[1] - seg_len
        total_len = h.shape[1]
        
        # compute projections of input and memory embeddings
        q = self.w_q(x).view(batch_size, -1, seg_len, embed_dim)
        kv = self.w_kv(h).view(2*batch_size, -1, total_len, embed_dim)
        k, v = kv.chunk(2, dim=0)
        k = k.transpose(2, 3) # only using transposed k below
        
        # relative distance between two tokens is max total_len
        pos = self.pos[-total_len:]
        r = self.w_r(pos).view(-1, total_len, embed_dim)
        
        # compute relative positional encodings
        b = q @ r.transpose(1, 2)
        b = self.circulant_shift(b, -seg_len+1)
        
        # u1 and u2 should be identical for all query vectors
        u1 = self.u1.repeat(1, seg_len, 1)
        u2 = self.u2.repeat(1, seg_len, 1)
        
        # this is the XL specific way of computing the attention score
        #att_mask = att_mask.unsqueeze(1).unsqueeze(-1).repeat(1,10,1,total_len)
        att_score = q @ k + b + u1 @ k + u2 @ r.transpose(1, 2)
        #att_score = att_score * att_mask
        att_score = att_score.tril(mem_len) / embed_dim**0.5
        att_score[att_score == 0] = float("-inf")
        att_score = torch.softmax(att_score, dim=-1)
        
        # compute output and save to memory
        att = (att_score @ v).view(batch_size, seg_len, -1)
        return self.layer_norm(self.mlp(att) + x)
          
    def get_sinusoid_pos_encoding(self, mem_len, embed_dim):
        """
        Standard sinusoid positional encoding method outlined in the original
        Transformer paper. In this case, we use the encodings not to represent
        each token's position in a sequence but to represent the distance
        between two tokens (i.e. as a *relative* positional encoding).
        """
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