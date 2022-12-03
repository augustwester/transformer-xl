class Config:
    def __init__(self,
                 model_dim=None,
                 embed_dim=None,
                 seg_len=None,
                 mem_len=None,
                 num_heads=None,
                 dropout=None,
                 inner_dim=None,
                 num_layers=None):
        
        self.model_dim = model_dim
        self.embed_dim = embed_dim
        self.seg_len = seg_len
        self.mem_len = mem_len
        self.num_heads = num_heads
        self.dropout = dropout
        self.inner_dim = inner_dim
        self.num_layers = num_layers