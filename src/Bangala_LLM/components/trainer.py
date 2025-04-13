import torch
from Bangala_LLM.components.data_loader import create_dataloader_v1
from Bangala_LLM.components.models import GPT2
from Bangala_LLM.utils.common import (
    calc_loss_loader,
    calc_loss_batch,
    generate,
    token_indx_to_text,
    text_to_token_indx,
)
import tiktoken
import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "num_heads": 12,
    "dropout": 0.1,
    "num_layers": 12,
    "qkv_bias": False,
}


model = GPT2(GPT_CONFIG_124M).to(device)
torch.manual_seed(123)  # For reproducibility due to the shuffling in the data loader


trai_data_path = "/home/amzad/Desktop/bangla_GPT/dataset/test.txt"
val_data_path = "/home/amzad/Desktop/bangla_GPT/dataset/prothom_alo.txt"

# Load the train and validation data
with open(trai_data_path, "r") as f:
    train_data = f.read()


with open(val_data_path, "r") as f:
    val_data = f.read()


val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=4,
)

train_loader = create_dataloader_v1(
    train_data,
    batch_size=1,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=4,
)

print("data loaded")


def train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq,
    eval_iter,
    start_context,
    tokenizer,
):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # Calculate loss gradients
            optimizer.step()  # Update model weights using loss gradients
            tokens_seen += input_batch.numel()
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(
                    f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                )

        # Print a sample text after each epoch
                generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_indx(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate(
            model=model, idx=encoded, max_new_tokens=500, context_size=context_size
        )
    decoded_text = token_indx_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()


optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)


tokenizer = tiktoken.get_encoding("gpt2")
num_epochs = 1
train_losses, val_losses, tokens_seen = train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs=num_epochs,
    eval_freq=100,
    eval_iter=500,
    start_context="<p>গাইবান্ধার মৃত্যুদণ্ডপ্রাপ্ত সাবেক সংসদ সদস্য কর্নেল কাদের খান মারা গেছেন -",
    tokenizer=tokenizer,
)


class trainner:
    def __init__(
        self, config, model, train_loader, val_loader, optimizer, device, tokenizer
    ):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.start_context = (
            "গাইবান্ধার মৃত্যুদণ্ডপ্রাপ্ত সাবেক সংসদ সদস্য কর্নেল কাদের খান মারা গেছেন -"
        )
        self.tokenizer = tokenizer

    def train_model_simple(self, num_epochs, eval_freq, eval_iter):

        # Initialize lists to track losses and tokens seen
        train_losses, val_losses, track_tokens_seen = [], [], []
        tokens_seen, global_step = 0, -1

        # Main training loop
        pbar = tqdm.tqdm(
            enumerate(train_loader), total=len(train_loader)
        )  # Progress bar

        for epoch in range(num_epochs):  # Epoch loop
            self.model.train()

            for i, (input_batch, target_batch) in pbar:  # Batch loop
                self.optimizer.zero_grad()
                loss = calc_loss_batch(
                    input_batch, target_batch, self.model, self.device
                )
                loss.backward()
                self.optimizer.step()
                tokens_seen += input_batch.numel()
                global_step += 1

                # Optional evaluation step
                if global_step % eval_freq == 0:
                    train_loss, val_loss = evaluate_model(
                        model, train_loader, val_loader, device, eval_iter
                    )
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)
                    print(
                        f"Ep {epoch+1} (Step {global_step:06d}): "
                        f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                    )

            # Print a sample text after each epoch
            generate_and_print_sample(model, tokenizer, device, start_context)

    def evaluate_model(self, eval_iter):
        self.model.eval()
        with torch.no_grad():
            train_loss = calc_loss_loader(
                self.train_loader, self.model, self.device, num_batches=eval_iter
            )
            val_loss = calc_loss_loader(
                self.val_loader, self.model, self.device, num_batches=eval_iter
            )
        self.model.train()
        return train_loss, val_loss

    def generate_and_print_sample(self):
        self.model.eval()
        context_size = self.model.pos_emb.weight.shape[0]
        encoded = text_to_token_indx(self.start_context, self.tokenizer).to(self.device)
        with torch.no_grad():
            token_ids = generate(
                model=self.model,
                idx=encoded,
                max_new_tokens=50,
                context_size=context_size,
            )
        decoded_text = token_indx_to_text(token_ids, self.tokenizer)
        print(decoded_text.replace("\n", " "))
        self.model.train()
