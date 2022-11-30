import torch
from model import TransformerXL
from data import gen_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
saved = torch.load("model.pt", map_location=torch.device("cpu"))
state_dict, config = saved["state_dict"], saved["config"]

model = TransformerXL(config, device)
model.load_state_dict(state_dict)
model.eval()

seq_len = 20
num_samples = 1000
test_data, test_targets = gen_dataset(0, 10, seq_len, num_samples)
preds = torch.empty(num_samples, 0)

for i in range(20):
    out = model(test_data) if i == 0 else model(next_token)
    next_token = out.argmax(-1)[:, -1:]
    preds = torch.cat((preds, next_token), dim=-1)

num_correct = (test_data.sort()[0] == preds).prod(1).sum()
print(f"{num_correct} / {num_samples}")