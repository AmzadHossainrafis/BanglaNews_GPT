import torch
import torch.nn as nn
from Bangala_LLM.components.helper import TransformerBlock


class GPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tokens_embeddings = nn.Embedding(config["vocab_size"], config["emb_dim"])
        self.pos_emb = nn.Embedding(config["context_length"], config["emb_dim"])
        self.dropout = nn.Dropout(config["dropout"])

        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config["num_layers"])]
        )

        self.final_norm = nn.LayerNorm(config["emb_dim"])  # LayerNorm
        self.out_head = nn.Linear(config["emb_dim"], config["vocab_size"], bias=False)

    def forward(self, x):
        batch, seq_len = x.size()
        token_emb = self.tokens_embeddings(x)
        pos_emb = self.pos_emb(torch.arange(seq_len, device=x.device))

        x = token_emb + pos_emb
        x = self.dropout(x)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


# if __name__ == "__main__":
#     import tiktoken

#     tokenizer = tiktoken.get_encoding("gpt2")
#     batch = []
#     text1 = "আমি বাংলায় গান গাই।"
#     text2 = "আমি বাংলায় গান গাই।"

#     batch.append(torch.tensor(tokenizer.encode(text1)))
#     batch.append(torch.tensor(tokenizer.encode(text2)))

#     batch = torch.stack(batch)

#     gpt_confg = {
#         "vocab_size": 50257,
#         "context_length": 1024,
#         "emb_dim": 768,
#         "num_heads": 12,
#         "dropout": 0.1,
#         "num_layers": 12,
#         "qkv_bias": False,
#     }

#     # model = DummyGPt(gpt_confg)
#     # out = model(batch)
#     # print(f"output shape: {out.size()}")
#     # print(out)
#     model = GPT2(gpt_confg)
#     out = model(batch)
#     print(f"output shape: {out.size()}")

# #     x = torch.randn(2, 10, 512)
# #     multihead = MHAttention(512, 512, 8, 10)
# #     out = multihead(x)
# #     print(out.size())


# # # if __name__ == '__main__':
# # #     x = torch.randn(2, 10, 512)
# # #     multihead = MultyHeadAttention(512, 512, 10, 8)
# # #     out = multihead(x)
# # #     print(out.size())
