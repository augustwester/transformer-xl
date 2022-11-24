from torch.nn import CrossEntropyLoss
from model import TransformerXL
from config import Config
from datasets.load import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from math import ceil
from torch.optim import Adam
from torch.nn.functional import one_hot
from tqdm import tqdm
from torch.distributions import Categorical
import torch

num_epochs = 5
batch_size = 32

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

# configure model hyperparameters
config = Config()
config.model_dim = 200
config.embed_dim = 200
config.seg_len = 100
config.mem_len = 384
config.num_heads = 2
config.dropout = 0
config.inner_dim = 200
config.vocab_size = tokenizer.vocab_size
config.num_layers = 2

# create model, dataset loader, and optimizer
model = TransformerXL(config)
train_loader = DataLoader(train, batch_size=batch_size)
opt = Adam(model.parameters(), lr=5e-4)

def generate_sentence(model, num_gen_words=10):
    input = "The adventure began when "
    for _ in range(num_gen_words):
        tokenized_input = tokenizer(input, return_tensors="pt")
        out = model(tokenized_input["input_ids"], tokenized_input["attention_mask"])
        next_token_dist = torch.softmax(out, dim=-1)[0, -1]
        next_token_id = Categorical(next_token_dist).sample()
        next_token = tokenizer.decode(next_token_id)
        input += next_token + " "
    model.clear_memory()
    return input

progress = tqdm(train_loader)
for _ in range(num_epochs):
    for batch_num, batch in enumerate(progress):
        x = batch["input_ids"]
        att_mask = batch["attention_mask"]
        num_segments = ceil(x.shape[-1] / config.seg_len)
        for i in range(num_segments):
            seg = x[:, i*config.seg_len:(i+1)*config.seg_len]
            seg_att_mask = att_mask[:, i*config.seg_len:(i+1)*config.seg_len]
            
            preds = model(seg, seg_att_mask)[seg_att_mask.bool()]
            targets = seg[seg_att_mask.roll(1).bool()]
            
            cross_entropy = CrossEntropyLoss(label_smoothing=0.1)
            loss = cross_entropy(preds, targets)
            loss.backward()
            opt.step()
            opt.zero_grad()
        progress.set_description(f"{loss.item()}")
        model.clear_memory()
        
        if batch_num % 20 == 0:
            print(generate_sentence(model))
            