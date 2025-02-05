import torch 
import tiktoken
def generate_text_simple(model , indx , max_new_token , context_size):


    for _ in range(max_new_token):
        with torch.no_grad():
            indx_cond = indx[: , -context_size:]
            with torch.no_grad():
                logits = model(indx_cond)

        logits = logits[: , -1 , :]
        prob = torch.softmax(logits , dim = -1)
        indx_next = torch.argmax(prob , dim = -1 , keepdim = True)
        indx = torch.cat([indx , indx_next] , dim = 1)

    return indx 


def text_to_token_indx(txt , tokenizer):
    token_ids = tokenizer.encode(txt, allowed_special='<|endoftext|>')
    return torch.tensor(token_ids).unsqueeze(0)


def token_indx_to_text(indx , tokenizer):
    flat = indx.squeeze().tolist()
    return tokenizer.decode(flat)








if __name__ == "__main__":

    from src.Bangala_LLM.components.models import  GPT2

    gpt_confg = {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 768,
        "num_heads": 12,
        "dropout": 0.1,
        "num_layers": 12,
        "qkv_bias": False,
    }

    model = GPT2(gpt_confg)
    indx = torch.tensor([[0, 50256]])
    out = generate_text_simple(model , indx , 10 , 10)
    print(out)
    print(out.size())