"""Cross-entropy kaybı.

Modelin (B, T, vocab_size) logits'i ile (B, T) hedeflerini karşılaştırır.
Hesaptan önce batch ve zaman eksenleri tek eksende düzleştirilir, çünkü
F.cross_entropy iki boyutlu logits bekler; her konum bağımsız bir
sınıflandırma örneği olarak ele alınır.

Softmax ayrıca uygulanmaz -- F.cross_entropy log-softmax'ı sayısal olarak
kararlı biçimde kendi içinde yapar.
"""

import torch.nn.functional as F
import torch


def compute_loss(logits, targets)->torch.Tensor:
    B, T, C = logits.shape
    logits = logits.view(B * T, C)
    targets = targets.view(B * T)
    loss = F.cross_entropy(logits, targets)
    return loss

# Manual test
# if __name__ == "__main__":
#     from model.gpt import GPT
#
#     demo_model = GPT(vocab_size=27,
#                         block_size=8,
#                         n_embd=16,
#                         n_head=4,
#                         n_layer=2)
#     fake_input = torch.randint(0, 27, (5, 7))
#     logits = demo_model(fake_input)
#     fake_targets = torch.randint(0, 27, (5, 7))
#     loss_value = compute_loss(logits, fake_targets)
#     print(loss_value)
