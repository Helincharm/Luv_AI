"""Uctan uca GPT dil modeli.

Token dizisini alır, gömme katmanından geçirir, n_layer adet Transformer
bloğunu sırayla uygular, son bir LayerNorm'dan sonra dil modeli başlığı ile
her pozisyon için sözlük boyutunda logits üretir.

Çıktı (B, T, vocab_size) şeklindedir: dizideki her konum için bir sonraki
tokenın olasılık dağılımı. Eğitimde bu dağılımların tamamı kullanılır,
üretimde yalnızca son konumdaki.
"""

import torch
import torch.nn as nn
from model.embeddings import Embeddings
from model.transformer_block import TransformerBlock

class GPT(nn.Module):
    def __init__(self,vocab_size, block_size, n_embd, n_head, n_layer)->None:
        super().__init__()
        self.embeddings = Embeddings(vocab_size, block_size, n_embd)
        block_list =[]

        for _ in range(n_layer):
            block_list.append(TransformerBlock(n_embd, n_head, block_size))

        self.blocks = nn.ModuleList(block_list)
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx)->torch.Tensor:
        x = self.embeddings(idx)

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

# Manual test
# if __name__ == "__main__":
#     demo_model = GPT(vocab_size = 27,
#     block_size = 8,
#     n_embd = 16,
#     n_head = 4,
#     n_layer = 2)
#     fake_input = torch.randint(0, 27, (5, 7))
#     output = demo_model(fake_input)
#     print(fake_input.shape, output.shape)
