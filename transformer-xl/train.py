import torch
from torch.nn import CrossEntropyLoss
from model import TransformerXL
from math import ceil
from torch.optim import Adam
from tqdm import tqdm
from data import gen_dataset
import argparse
from torch.optim.lr_scheduler import LinearLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(num_epochs, batch_size, num_samples, num_digits, seq_len):
    # generate dataset of random sequences of unordered digits
    train_data, train_targets = gen_dataset(0, num_digits, seq_len, num_samples)
    num_batches = ceil(num_samples / batch_size)

    # for sequence [4,5,3,0], we input [4,5,3,0,0,3,4] during training
    train_data = torch.cat((train_data, train_targets), dim=-1)[:, :-1]

    # set up model configuration
    config = TransformerXL.default_config()
    config.mem_len = 0 # we don't need memory during training for this task
    config.seg_len = 2 * seq_len - 1
    config.vocab_size = num_digits

    # create model along with optimizer and loss function
    model = TransformerXL(config, device)
    opt = Adam(model.parameters(), lr=1e-3)
    scheduler = LinearLR(opt, start_factor=1.0, end_factor=0.01, total_iters=num_epochs*num_batches)
    cross_entropy = CrossEntropyLoss()

    for _ in range(num_epochs):
        progress = tqdm(range(num_batches))
        for i in progress:
            x = train_data[i*batch_size:(i+1)*batch_size].to(device)
            y = train_targets[i*batch_size:(i+1)*batch_size].to(device)
            preds = model(x)[:, seq_len-1:].reshape(-1, num_digits)
            loss = cross_entropy(preds, y.flatten())
            loss.backward()
            opt.step()
            opt.zero_grad()
            scheduler.step()
            progress.set_description(f"{loss.item()}")
            model.clear_memory()

    # save model and model config to disk
    torch.save({"state_dict": model.state_dict(),
                "config": config}, "model.pt")
    print("Model trained and saved to file: model.pt.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--sequence_length", type=int, default=20)
    parser.add_argument("--num_digits", type=int, default=10)
    parser.add_argument("--num_samples", type=int, default=50000)

    args = parser.parse_args()
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    seq_len = args.sequence_length
    num_digits = args.num_digits
    num_samples = args.num_samples
    
    train(num_epochs, batch_size, num_samples, num_digits, seq_len)