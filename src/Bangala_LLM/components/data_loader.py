import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
from Bangala_LLM.utils.logger import logger


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length=4, stride=1):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    txt, batch_size, max_length, stride, shuffle=True, drop_last=True, num_workers=0
):
    # Initialize the tokenizer - using gpt2 to match model vocab_size
    tokenizer = tiktoken.get_encoding("gpt2")
    # tokenizer.add_special_tokens({"eos_token": "<|EOS|>"})
    logger.info("Tokenizer initialized and special tokens added.")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    logger.info(f"DataLoader created with {len(dataloader)} batches.")
    return dataloader


# if __name__ == "__main__":
#     sample_text = "This is a sample text for testing the GPTDatasetV1 and DataLoader.<|endoftext|> It contains multiple sentences to ensure that the sliding window works correctly.<|endoftext|>"
#     dataloader = create_dataloader_v1(
#         sample_text, batch_size=2, max_length=5, stride=1, num_workers=3
#     )

#     print("Sample batches from DataLoader:")
#     for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
#         print(f"Batch {batch_idx}:")
#         print("Input IDs:", input_ids)
#         print("Target IDs:", target_ids)
#         if batch_idx == 10:  # Just show first 3 batches for brevity
#             break