import torch
from torch.nn import CrossEntropyLoss
from model import TransformerXL
from datasets.load import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from math import ceil
from torch.optim import Adam
from tqdm import tqdm
from torch.distributions import Categorical

num_epochs = 5
batch_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load dataset
dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
train = dataset["train"]
train = train.filter(lambda x: x["text"] != "" and x["text"][:2] != " =")
#train = train.filter(lambda x: "the" not in x["text"])

# tokenize dataset
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
def tokenization(example):
    return tokenizer(example["text"], padding="longest", max_length=-1)
train = train.map(tokenization, batched=True, batch_size=batch_size)
train.set_format(type="pt")

# create model, dataset loader, and optimizer
config = TransformerXL.get_default_config()
config.vocab_size = tokenizer.vocab_size
model = TransformerXL(config, device)
train_loader = DataLoader(train, batch_size=batch_size)
opt = Adam(model.parameters(), lr=1e-4)
cross_entropy = CrossEntropyLoss(label_smoothing=0.1)

def generate_sentence(model, num_gen_words=10):
    output = "The adventure began when "
    input = output
    for _ in range(num_gen_words):
        tokenized_input = tokenizer(input, return_tensors="pt")
        out = model(tokenized_input["input_ids"].to(device),
                    tokenized_input["attention_mask"].to(device))
        next_token_dist = torch.softmax(out, dim=-1)[0, -2]
        next_token_id = Categorical(next_token_dist).sample()
        next_token = tokenizer.decode(next_token_id)
        output += next_token + " "
        input = next_token
    model.clear_memory()
    return output

progress = tqdm(train_loader)
for _ in range(num_epochs):
    for batch_num, batch in enumerate(progress):
        x = batch["input_ids"].to(device)
        att_mask = batch["attention_mask"].to(device)
        num_segments = ceil(x.shape[-1] / config.seg_len)
        
        for i in range(num_segments):
            seg = x[:, i*config.seg_len:(i+1)*config.seg_len]
            seg_att_mask = att_mask[:, i*config.seg_len:(i+1)*config.seg_len]
            preds = model(seg, seg_att_mask)[seg_att_mask.bool()]
            targets = seg[seg_att_mask.roll(1).bool()]
            loss = cross_entropy(preds, targets)
            loss.backward()
            opt.step()
            opt.zero_grad()
        progress.set_description(f"{loss.item()}")
        model.clear_memory()
        
        if batch_num % 20 == 0:
            print(generate_sentence(model))
            