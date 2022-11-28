import torch
from torch import nn
from decoder import DecoderLayer
from config import Config

class TransformerXL(nn.Module):
    @staticmethod
    def get_default_config():
        config = Config()
        config.model_dim = 200
        config.embed_dim = 200
        config.seg_len = 100
        config.mem_len = 384
        config.num_heads = 2
        config.dropout = 0
        config.inner_dim = 200
        config.num_layers = 2
        return config
    
    def __init__(self, config, device):
        super().__init__()
        self.mem = None
        self.mem_len = config.mem_len
        self.model_dim = config.model_dim
        self.vocab_size = config.vocab_size
        self.embed = nn.Embedding(config.vocab_size, config.model_dim)
        self.layers = nn.ModuleList()
        self.device = device
        
        # no need for each layer to create a copy of the positional encodings
        total_len = self.mem_len+config.seg_len
        R = self.get_sinusoid_pos_encoding(total_len, self.model_dim)
        R = torch.flip(R, dims=(0,)).to(device)
        
        for _ in range(config.num_layers):
            dec = DecoderLayer(config.model_dim,
                               config.embed_dim,
                               config.mem_len,
                               config.num_heads,
                               config.inner_dim,
                               R,
                               device)
            self.layers.append(dec)
            
        self.out_layer = nn.Linear(config.model_dim, config.vocab_size)
        self.to(device)
    
    def forward(self, x, att_mask):
        x = self.embed(x)
        
        # create memory tensors if they haven't been already
        if self.mem is None:
            batch_size = x.size(0)
            self.set_up_memory(batch_size)
        
        # compute model output, saving layer inputs to memory along the way
        for i, dec in enumerate(self.layers):
            x_ = x.clone()
            x = dec(x, self.mem[i], att_mask)
            self.add_to_memory(x_, i)
            
        return self.out_layer(x)
    
    def get_sinusoid_pos_encoding(self, total_len, embed_dim):
        """
        Standard sinusoid positional encoding method outlined in the original
        Transformer paper. In this case, we use the encodings not to represent
        each token's position in a sequence but to represent the distance
        between two tokens (i.e. as a *relative* positional encoding).
        """
        pos = torch.arange(total_len).unsqueeze(1)
        enc = torch.arange(embed_dim).float()
        enc = enc.unsqueeze(0).repeat(total_len, 1)
        enc[:, ::2] = torch.sin(pos / 10000**(2*enc[:, ::2]/embed_dim))
        enc[:, 1::2] = torch.cos(pos / 10000**(2*enc[:, 1::2]/embed_dim))
        return enc
    
    def set_up_memory(self, batch_size):
        self.mem = [None] * len(self.layers)
        for i in range(len(self.mem)):
            self.mem[i] = torch.zeros(batch_size, 0, self.model_dim).to(self.device)
    
    def add_to_memory(self, x, i):
        # beware that this indexing might cause problems when mem_len==0
        self.mem[i] = torch.cat((self.mem[i], x.detach()), dim=1)[:, -self.mem_len:]
    
    def clear_memory(self):
        self.mem = None