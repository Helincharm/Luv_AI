"""Optimizer kurulumu.

Model parametreleri için AdamW üretir. AdamW ağırlık sönümünü (weight decay)
gradyan güncellemesinden ayırır, böylece düzenlileştirmeyi öğrenme oranından
bağımsız uygular; Transformer eğitiminde fiili standart budur.
"""

import torch

def create_optimizer(model, learning_rate) -> torch.optim.Optimizer:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    return optimizer

# Manual test
# if __name__ == "__main__":
#     from model.gpt import GPT
#
#     demo_model = GPT(vocab_size=27,
#                      block_size=8,
#                      n_embd=16,
#                      n_head=4,
#                      n_layer=2)
#     demo_optimizer = create_optimizer(demo_model, learning_rate=0.001)
#     print(demo_optimizer)
