import torch

def gen_dataset(low, high, seq_len, num_samples):
    inputs = torch.randint(low, high, size=(num_samples, seq_len))
    targets, _ = inputs.sort()
    return inputs, targets