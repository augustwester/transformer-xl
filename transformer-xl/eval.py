import torch
import argparse
from model import TransformerXL
from data import gen_dataset
from tqdm import tqdm
from math import ceil
from time import time

def eval(seq_len, num_digits, num_samples, batch_size):
    # load model from disk
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load("model.pt", map_location=device)
    state_dict, config = saved["state_dict"], saved["config"]

    model = TransformerXL(config, device)
    model.load_state_dict(state_dict)
    model.eval()

    # generate test dataset
    test_data, test_targets = gen_dataset(0, num_digits, seq_len, num_samples)
    test_data, test_targets = test_data.to(device), test_targets.to(device)
    num_batches = ceil(num_samples / batch_size)
    
    print("Evaluating without memory...")

    # first test the model without using the hidden state memory
    model.mem_len = 0
    preds = torch.empty(0, seq_len).to(device)
    start = time()
    
    for i in tqdm(range(num_batches)):
        batch_preds = test_data[i*batch_size:(i+1)*batch_size]
        for _ in range(seq_len):
            out = model(batch_preds)
            next_token = out.argmax(-1)[:, -1:]
            batch_preds = torch.cat((batch_preds, next_token), dim=-1)
        model.clear_memory()
        preds = torch.cat((preds, batch_preds[:, -seq_len:]))
        
    end = time()
    num_correct = (test_data.sort()[0] == preds[:, -seq_len:]).prod(1).sum()
    print(f"Achieved accuracy of {num_correct / num_samples} in {end - start} seconds\n")

    print("Evaluating with memory...")
    
    # then we test the model using the hidden state memory
    model.mem_len = seq_len * 2 - 1
    preds = torch.empty(num_samples, seq_len).to(device)
    start = time()
    
    for i in tqdm(range(num_batches)):
        seqs = test_data[i*batch_size:(i+1)*batch_size]
        for j in range(seq_len):
            out = model(seqs) if j == 0 else model(next_token)
            next_token = out.argmax(-1)[:, -1:]
            preds[i*batch_size:(i+1)*batch_size, j:j+1] = next_token
        model.clear_memory()
        
    end = time()
    num_correct = (test_data.sort()[0] == preds).prod(1).sum()
    print(f"Achieved accuracy of {num_correct / num_samples} in {end - start} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence_length", type=int, default=20)
    parser.add_argument("--num_digits", type=int, default=10)
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=128)

    args = parser.parse_args()
    seq_len = args.sequence_length
    num_digits = args.num_digits
    num_samples = args.num_samples
    batch_size = args.batch_size
    
    eval(seq_len, num_digits, num_samples, batch_size)
