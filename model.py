import torch
from torch import nn
from decoder import DecoderLayer

class TransformerXL(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mem = [None] * config.num_layers
        self.mem_len = config.mem_len
        self.model_dim = config.model_dim
        self.vocab_size = config.vocab_size
        self.embed = nn.Embedding(config.vocab_size, config.model_dim)
        self.layers = nn.ParameterList()
        
        for _ in range(config.num_layers):
            dec = DecoderLayer(config.model_dim,
                               config.embed_dim,
                               config.seg_len,
                               config.mem_len,
                               config.num_heads,
                               config.inner_dim)
            self.layers.append(dec)
            
        self.out_layer = nn.Linear(config.model_dim, config.vocab_size)
    
    def forward(self, x, att_mask):
        x = self.embed(x)
        for i, dec in enumerate(self.layers):
            if self.mem[i] is None:
                batch_size = x.shape[0]
                self.mem[i] = torch.zeros(batch_size, self.mem_len, self.model_dim)
            x = dec(x, self.mem[i], att_mask)
            # beware that this indexing might cause problems when mem_len==0
            self.mem[i] = torch.cat((self.mem[i], x.detach()), dim=1)[:, -self.mem_len:]
        return self.out_layer(x)
    
    def clear_memory(self):
        self.mem = [None] * len(self.layers)