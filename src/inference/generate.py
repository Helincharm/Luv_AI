"""Otoregresif metin üretimi.

Verilen bir başlangıç metninden itibaren token token yeni metin üretir: model
son konumun logits'ini verir, sampler bir token seçer, seçilen token girdinin
sonuna eklenir ve döngü tekrarlar.

Bağlam block_size'ı aştığında baştan kırpılır -- model, pozisyon gömme tablosu
kadar uzun bir diziden fazlasını göremez.
"""

import torch
from inference.sampler import sample_next_token


def generate(model, idx, max_new_tokens, block_size, temperature=1.0) -> torch.Tensor:
    model.eval()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            context_tokens = idx[:, -block_size:]

            logits = model(context_tokens)

            next_token_logits = logits[:, -1, :]

            next_token = sample_next_token(next_token_logits, temperature=temperature)

            idx = torch.cat((idx, next_token), dim=1)

    return idx


# Manual test
# if __name__ == '__main__':
#
#     from model.gpt import GPT
#
#     demo_model = GPT(vocab_size=27,
#                      block_size=8,
#                      n_embd=16,
#                      n_head=4,
#                      n_layer=2)
#
#     fake_input = torch.randint(0,27,(1,5))
#
#     generated_idx = generate(demo_model,fake_input, max_new_tokens=10, block_size=8)
#
#     print(generated_idx.shape)


