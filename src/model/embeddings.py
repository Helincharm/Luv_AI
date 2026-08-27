"""Token ve pozisyon gömme katmanı.

Her token kimliğini öğrenilebilir bir vektöre çevirir ve üzerine o tokenın
dizideki sırasını temsil eden ikinci bir vektör ekler.

Pozisyon bilgisinin ayrıca eklenmesi zorunludur: self-attention girdiyi
sırasız bir küme gibi işler ve tek başına "önce/sonra" ayrımını göremez.
"""

import torch
import torch.nn as nn


class Embeddings(nn.Module):
    def __init__(self, vocab_size: int, block_size: int, n_embd: int) -> None:

        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, n_embd)

        self.position_embedding = nn.Embedding(block_size, n_embd)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:

        tok_emb = self.token_embedding(idx)

        seq_len = idx.shape[1]
        positions = torch.arange(seq_len, device=idx.device)

        pos_emb = self.position_embedding(positions)

        return tok_emb + pos_emb
