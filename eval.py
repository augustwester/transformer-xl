import torch
from model import TransformerXL

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

saved = torch.load("model.pt")
state_dict, config = saved["state_dict"], saved["config"]
config.seq_len = 1
config.mem_len = 20

model = TransformerXL(config, device)
model.load_state_dict(state_dict)
model.eval()

num_samples = 100
seqs = torch.randint(0, 10, size=(num_samples, 20)).to(device)
preds = torch.empty(num_samples, 0)

for i in range(20):
    out = model(seqs) if i == 0 else model(next_token)
    next_token = out.argmax(-1)[:, -1:]
    preds = torch.cat((preds, next_token), dim=-1)
    
num_correct = (seqs == preds).prod(1).sum()
print(f"{num_correct} / {num_samples}")

