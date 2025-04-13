import torch
from Bangala_LLM.components.data_loader import create_dataloader_v1
from Bangala_LLM.components.models import GPT2

import tiktoken
import tqdm

tokenizer = tiktoken.get_encoding("gpt2")

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "num_heads": 12,
    "dropout": 0.1,
    "num_layers": 12,
    "qkv_bias": False,
    "train_data_path": "/home/amzad/Desktop/bangla_GPT/dataset/test.txt",
    "val_data_path": "/home/amzad/Desktop/bangla_GPT/dataset/prothom_alo.txt",
    "batch_size": 1,
    "lr": 0.0004,
    "weight_decay": 0.1,
    "num_epochs": 10,
    "eval_freq": 100,
    "seed": 123,
    "max_new_tokens": 500,
    "start_context": "গাইবান্ধার মৃত্যুদণ্ডপ্রাপ্ত সাবেক সংসদ সদস্য কর্নেল কাদের খান মারা গেছেন -",
    "num_workers": 4,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}


model = GPT2(GPT_CONFIG_124M).to(GPT_CONFIG_124M["device"])
torch.manual_seed(123)  #

with open(GPT_CONFIG_124M["train_data_path"], "r") as f:
    train_data = f.read()


with open(GPT_CONFIG_124M["val_data_path"], "r") as f:
    val_data = f.read()


val_loader = create_dataloader_v1(
    val_data,
    batch_size=GPT_CONFIG_124M["batch_size"],
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=GPT_CONFIG_124M["num_workers"],
)

train_loader = create_dataloader_v1(
    train_data,
    batch_size=GPT_CONFIG_124M["batch_size"],
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=GPT_CONFIG_124M["num_workers"],
)
print("data loaded")
