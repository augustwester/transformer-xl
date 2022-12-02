import torch
from torch.nn import CrossEntropyLoss
from model import TransformerXL
from math import ceil
from torch.optim import Adam
from tqdm import tqdm
from data import gen_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 16
batch_size = 128
num_samples = 10**5
vocab_size = 10
seq_len = 20
num_batches = ceil(num_samples / batch_size)

# generate dataset of random sequences of unordered digits
train_data, train_targets = gen_dataset(0, vocab_size, seq_len, num_samples)

# for sequence [4,5,3,0], we input [4,5,3,0,0,3,4] during training
train_data = torch.cat((train_data, train_targets), dim=-1)[:, :-1]

# set up model configuration
config = TransformerXL.get_default_config()
config.mem_len = 0 # we don't need memory during training for this task
config.seg_len = 2 * seq_len - 1
config.vocab_size = vocab_size

# create model along with optimizer and loss function
model = TransformerXL(config, device)
opt = Adam(model.parameters(), lr=1e-4)
cross_entropy = CrossEntropyLoss()

for _ in range(num_epochs):
    progress = tqdm(range(num_batches))
    for i in progress:
        x = train_data[i*batch_size:(i+1)*batch_size].to(device)
        y = train_targets[i*batch_size:(i+1)*batch_size].to(device)
        preds = model(x)[:, seq_len-1:].reshape(-1, vocab_size)
        loss = cross_entropy(preds, y.flatten())
        loss.backward()
        opt.step()
        opt.zero_grad()
        progress.set_description(f"{loss.item()}")
        model.clear_memory()

# save model and model config to disk
torch.save({"state_dict": model.state_dict(),
            "config": config}, "model.pt")
print("Model trained and saved to file: model.pt. Proceeding to evaluate...")

# evaluate model using single-token autoregression
num_test_samples = 1000
test_data, test_targets = gen_dataset(0, vocab_size, seq_len, num_test_samples)
preds = torch.empty(num_test_samples, 0)

# on first run, input unordered sequence, then only model's predictions
for i in tqdm(range(seq_len)):
    out = model(test_data) if i == 0 else model(next_token)
    next_token = out.argmax(-1)[:, -1:]
    preds = torch.cat((preds, next_token), dim=-1)

num_correct = (test_data.sort()[0] == preds).prod(1).sum()
print(f"{num_correct} / {num_samples}")
