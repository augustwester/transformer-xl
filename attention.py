import torch
from torch import nn
        
class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, embed_dim, mem_len, num_heads, R, device):
        super().__init__()
        
        self.R = R
        self.mem_len = mem_len
        self.embed_dim = embed_dim
        self.device = device
        
        self.u1 = nn.Parameter(torch.Tensor(1, num_heads, 1, embed_dim))
        self.u2 = nn.Parameter(torch.Tensor(1, num_heads, 1, embed_dim))
        
        self.w_q = nn.Linear(model_dim, num_heads*embed_dim, bias=False)
        self.w_kv = nn.Linear(model_dim, 2*num_heads*embed_dim, bias=False)
        self.w_r = nn.Linear(model_dim, num_heads*embed_dim, bias=False)
        self.mlp = nn.Linear(num_heads*embed_dim, model_dim, bias=False)
        self.layer_norm = nn.LayerNorm(model_dim)
    
    def forward(self, x, mem, att_mask):
        # concat output from previous layer with "memory" from earlier segments
        h = torch.cat((mem, x), dim=1)
        
        batch_size, seg_len, embed_dim = x.shape
        mem_len = h.shape[1] - seg_len
        total_len = h.shape[1]
        
        # compute projections of input and memory embeddings
        q = self.w_q(x).view(batch_size, -1, seg_len, embed_dim)
        kv = self.w_kv(h).view(2*batch_size, -1, total_len, embed_dim)
        r_emb = self.w_r(self.R[-total_len:]).view(1, -1, total_len, embed_dim)
        k, v = kv.chunk(2, dim=0)
        
        # the "XL specific" way of computing the pre-softmax attention score
        AC = torch.einsum("bhid,bhjd->bhij", q + self.u1, k)
        BD = torch.einsum("bhid,bhjd->bhij", q + self.u2, r_emb)
        BD = self.circulant_shift(BD, -seg_len+1)
        
        # computing the attention scores
        att_score = AC + BD
        att_score = att_score.tril(mem_len) / embed_dim**0.5
        att_score[att_score == 0] = float("-inf")
        att_score = torch.softmax(att_score, dim=-1)
        
        # compute output
        att = (att_score @ v).view(batch_size, seg_len, -1)
        return self.layer_norm(self.mlp(att) + x)
              
    def circulant_shift(self, x, shift):
        """
        Shifts top row of `x` by `shift`, second row by `shift-1`, etc. This is
        used to compute the relative positional encoding matrix in linear time
        (as opposed to quadratic time for the naive solution). Note: Right-hand
        side values are not zeroed out here.
        
        See Appendix B of the Transformer-XL paper for more details.
        """
        batch_size, num_heads, height, width = x.shape
        i = torch.arange(width).roll(shift).unsqueeze(0).to(self.device)
        i = i.flip(1).repeat(1, 2)
        i = i.unfold(dimension=1, size=width, step=1).flip(-1).unsqueeze(0)
        i = i.repeat(batch_size, num_heads, 1, 1)[:, :, :height]
        return x.gather(3, i)