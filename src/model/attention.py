"""Nedensel (causal) self-attention: tek başlık ve çok başlıklı hali.

Her token, query/key/value dönüşümleriyle bağlamdaki diğer tokenlara bakar ve
onlardan ağırlıklı bilgi toplar. Benzerlik puanları head_size'ın karekörüne
bölünerek ölçeklenir; bu olmadan softmax büyük boyutlarda doyuma gider ve
gradyanlar kaybolur.

Üst üçgen maske, bir tokenın kendisinden sonraki tokenları görmesini engeller.
Dil modelinin geleceği kopya çekmeden tahmin etmesini sağlayan kısıt budur.

Çok başlıklı yapıda n_embd boyutu başlıklara eşit bölünür; her başlık farklı
türde ilişki öğrenir, çıktılar birleştirilip tek bir doğrusal katmandan
geçirilir.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionHead(nn.Module):
    def __init__(self, n_embd, head_size, block_size) ->None:
        super(). __init__ ()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.head_size = head_size
        self.register_buffer("tril",torch.tril(torch.ones(block_size, block_size)))
        # True yapıldığında her forward'da attention ağırlıkları
        # last_attention_weights'a kopyalanır; modelin neye baktığını
        # incelemek için. Eğitimde kapalı kalır, aksi halde her adımda
        # gereksiz bellek harcar.
        self.record_attention = False
        self.last_attention_weights = None
    def forward(self, x) -> torch.Tensor:
        q=self.query(x)
        k=self.key(x)
        v=self.value(x)
        scores = q @ k.transpose(-2, -1)
        scores = scores / (self.head_size ** 0.5)
        seq_len = x.size(1)
        scores = scores.masked_fill(self.tril[:seq_len, :seq_len]==0,float('-inf'))
        attention_weights = F.softmax(scores,dim=-1)
        if self.record_attention:
            self.last_attention_weights = attention_weights.detach()
        out = torch.matmul(attention_weights, v)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size) ->None:
        super(). __init__ ()
        head_size = n_embd // n_head
        head_list = []
        for _ in range(n_head):
            head_list.append(AttentionHead(n_embd, head_size, block_size))
        self.heads = nn.ModuleList(head_list)

        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x) -> torch.Tensor:
        head_outputs = []
        for head in self.heads:
            head_outputs.append(head(x))
        combined = torch.cat(head_outputs,dim=-1)
        out = self.proj(combined)
        return out


# if __name__ == "__main__":
#     demo_model = MultiHeadAttention(n_embd=16, n_head=4, block_size=8)
#     fake_input = torch.randn(2, 8, 16)
#     output = demo_model(fake_input)
#     print(fake_input.shape, output.shape)
