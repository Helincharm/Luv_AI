"""Konum bazlı ileri beslemeli (feed-forward) katman.

Attention'dan gelen bilgiyi her token için bağımsız olarak işler:
n_embd -> 4*n_embd -> n_embd. Aradaki genişleme ve ReLU doğrusal olmayanlığı,
attention'ın tek başına sağlayamadığı temsil kapasitesini modele katar.

Attention tokenlar arasında bilgi taşır; bu katman taşınan bilgiyi tokenın
kendi içinde dönüştürür.
"""

import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.ReLU() , nn.Linear(4 * n_embd, n_embd))

    def forward (self, x) ->torch.Tensor:
        return self.net(x)


# Manual test
# if __name__ == "__main__":
#     demo_layer = FeedForward(n_embd = 16)
#     fake_input = torch.randn(2, 8, 16)
#     print(demo_layer(fake_input))
