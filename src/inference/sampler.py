"""Logits'ten bir sonraki tokenın seçilmesi.

Logits'i sıcaklığa böler, softmax ile olasılığa çevirir ve bu dağılımdan
örnekleme yapar.

Sıcaklık dağılımın keskinliğini ayarlar: 1'in altı dağılımı keskinleştirip
üretimi daha kararlı ama tekrara yatkın, 1'in üstü ise düzleştirip daha
çeşitli ama tutarsız hale getirir. Argmax yerine örnekleme kullanılır; aksi
halde model aynı girdiye hep aynı ve kısa sürede döngüye giren cevabı verir.
"""

import torch
import torch.nn.functional as F

def sample_next_token(logits, temperature=1.0):
    new_logits = logits / temperature
    probs = F.softmax(new_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


# Manual test
# if __name__ == "__main__":
#     from model.gpt import GPT
#
#     demo_model = GPT(vocab_size = 27,
#                         block_size = 8,
#                         n_embd = 16,
#                         n_head = 4,
#                         n_layer = 2)
#     fake_input = torch.randint(0, 27, (5, 7))
#     output = demo_model(fake_input)
#     last_position = (output[:, -1, :])
#     next_token = sample_next_token(last_position)
#     print(next_token)
