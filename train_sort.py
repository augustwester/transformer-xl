import torch
from torch.nn import CrossEntropyLoss
from model import TransformerXL
from math import ceil
from torch.optim import Adam
from tqdm import tqdm

num_epochs = 16
batch_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def gen_dataset(low, high, seq_len, num_samples):
    inputs = torch.randint(low, high, size=(num_samples, seq_len))
    targets, _ = inputs.sort()
    inputs = torch.cat((inputs, targets), dim=-1)[:, :-1]
    return inputs, targets

num_samples = 100000
vocab_size = 10
seq_len = 20
batch_size = 128

train_data, train_targets = gen_dataset(0, vocab_size, seq_len, num_samples)
config = TransformerXL.get_default_config()
config.seq_len = seq_len*2-1
config.vocab_size = vocab_size
config.dropout = 0

model = TransformerXL(config, device)
opt = Adam(model.parameters(), lr=1e-4)
cross_entropy = CrossEntropyLoss()

for _ in range(num_epochs):
    num_batches = ceil(num_samples / batch_size)
    progress = tqdm(range(num_batches))
    for i in progress:
        x = train_data[i*batch_size:(i+1)*batch_size].to(device)
        y = train_targets[i*batch_size:(i+1)*batch_size].to(device)
        preds = model(x)[:, seq_len-1:]
        
        loss = cross_entropy(preds.reshape(-1, vocab_size), y.flatten())
        loss.backward()
        
        opt.step()
        opt.zero_grad()
            
        progress.set_description(f"{loss.item()}")
        model.clear_memory()
        
        if i % 50 == 0:
            inp = x[:, :seq_len]
            pre = preds.argmax(-1)
            tru = x[:, :seq_len].sort()[0]
            cor = (pre == tru).prod(1).sum()
            print("Input:       ", x[0, :seq_len])
            print("Prediction:  ", preds.argmax(-1)[0])
            print("Ground truth:", x[0, :seq_len].sort()[0])
            print(f"Accuracy:     {cor} / {len(inp)}",)
            