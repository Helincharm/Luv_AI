"""Tek bir Transformer bloğu: attention + feed-forward.

Her alt katman pre-norm düzeninde uygulanır (önce LayerNorm, sonra katman) ve
çıktısı girdiye artık (residual) bağlantıyla eklenir.

Pre-norm tercihi derinlik içindir: normalizasyon artık yolunun dışında
kaldığı için gradyan bloklar boyunca bozulmadan akar ve derin yığınlar ek
ısınma (warmup) hilelerine gerek kalmadan kararlı eğitilir.
"""

import torch
import torch.nn as nn
from model.attention import MultiHeadAttention
from model.feed_forward import FeedForward

class TransformerBlock(nn.Module):

    def __init__ (self, n_embd, n_head, block_size):
        super().__init__()
        self.attention = MultiHeadAttention(n_embd, n_head, block_size)
        self.feedforward = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x)-> torch.Tensor:
        x = x + self.attention(self.ln1(x))
        x = x + self.feedforward(self.ln2(x))
        return x

# Manual test
# if __name__ == "__main__":
#     n_embd = 16
#     n_head = 4
#     block_size = 8
#     model = TransformerBlock(n_embd, n_head, block_size)
#     fake_input = torch.randn(2, 8, 16)   # (batch, seq_len, n_embd)
#     output = model(fake_input)
#     print(output.shape)
