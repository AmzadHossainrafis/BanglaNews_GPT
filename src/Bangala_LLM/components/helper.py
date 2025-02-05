import torch
import torch.nn as nn



class SelfAttention(nn.Module):
    def __init__(self, in_dim, d_out, context_length, dropout=0.1, qkv_bias=False):
        super().__init__()

        self.d_out = d_out
        self.W_query = nn.Linear(in_dim, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(in_dim, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(in_dim, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask", torch.tril(torch.ones(context_length, context_length), diagonal=-1)
        )

    def forward(self, x):
        batch, num_tokens, d_in = x.size()
        query = self.W_query(x)
        key = self.W_key(x)
        value = self.W_value(x)

        atten_scores = query @ key.transpose(-1, -2)
        atten_scores.masked_fill_(
            self.mask[:num_tokens, :num_tokens] == 0, float("-inf")
        )
        atten_weights = torch.nn.functional.softmax(
            atten_scores / key.size(-1) ** 0.5, dim=-1
        )
        atten_weights = self.dropout(atten_weights)

        out = atten_weights @ value
        return out


class MultyHeadAttention(nn.Module):
    def __init__(self, in_dim, d_out, context_length, num_heads, dropout=0.1):
        super().__init__()

        self.head = nn.ModuleList(
            [
                SelfAttention(in_dim, d_out, context_length, dropout)
                for _ in range(num_heads)
            ]
        )

    def forward(self, x):
        return torch.cat([h(x) for h in self.head], dim=-1)


# efficient multihead attention
class MHAttention(nn.Module):
    def __init__(
        self, in_dim, d_out, num_heads, context_length, qkv_bias=False, dropout=0.1
    ):
        super().__init__()
        assert d_out % num_heads == 0, "d_out should be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(in_dim, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(in_dim, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(in_dim, d_out, bias=qkv_bias)
        self.out_projection = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.tril(torch.ones(context_length, context_length), diagonal=-1)
        )

    def forward(self, x):
        batch, num_tokens, d_in = x.size()
        keys = (
            self.W_key(x)
            .view(batch, num_tokens, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        queries = (
            self.W_query(x)
            .view(batch, num_tokens, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        values = (
            self.W_value(x)
            .view(batch, num_tokens, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        atten_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        atten_scores.masked_fill_(mask_bool, float("-inf"))
        atten_weights = torch.nn.functional.softmax(
            atten_scores / keys.size(-1) ** 0.5, dim=-1
        )
        atten_weights = self.dropout(atten_weights)
        context = (
            (atten_weights @ values)
            .transpose(1, 2)
            .contiguous()
            .view(batch, num_tokens, self.d_out)
        )
        return self.out_projection(context)


class DummyGPt(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.token_emb = nn.Embedding(config["vocab_size"], config["emb_dim"])
        self.position_emb = nn.Embedding(config["context_length"], config["emb_dim"])
        self.dropout = nn.Dropout(config["dropout"])
        # transformer blocks

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config["num_layers"])]
        )
        self.final_norm = DummyNorm(config["emb_dim"])
        self.out_head = nn.Linear(config["emb_dim"], config["vocab_size"])

    def forward(self, x):

        batch, seq_len = x.size()
        token_emb = self.token_emb(x)
        position_emb = self.position_emb(torch.arange(seq_len, device=x.device))

        x = token_emb + position_emb
        x = self.dropout(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)

        logits = self.out_head(x)
        return logits


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = MHAttention(
            config["emb_dim"],
            config["emb_dim"],
            config["num_heads"],
            config["context_length"],
            config["qkv_bias"],
            config["dropout"],
        )
        self.ff = FeedForward(config)
        self.norm1 = DummyNorm(config["emb_dim"])
        self.norm2 = DummyNorm(config["emb_dim"])
        self.dropout = nn.Dropout(config["dropout"])

    def forward(self, x):
        res = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.dropout(x)
        x = res + x

        res = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout(x)
        x = res + x

        return x


class DummyNorm(nn.Module):
    def __init__(self, norm_dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(norm_dim))
        self.shift = nn.Parameter(torch.zeros(norm_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.scale * (x - mean) / (std + self.eps) + self.shift


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ff_layer = nn.Sequential(
            nn.Linear(config["emb_dim"], config["emb_dim"] * 4),
            GELU_Activation(),
            nn.Linear(config["emb_dim"] * 4, config["emb_dim"]),
        )

    def forward(self, x):
        return self.ff_layer(x)


class GELU_Activation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    (torch.sqrt(torch.tensor(2 / torch.pi)) * (x + 0.044715 * x**3))
                )
            )
        )

