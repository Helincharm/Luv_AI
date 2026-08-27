"""Eğitim batch'lerinin üretimi.

Uzun token dizisinden rastgele başlangıç noktaları seçer; her biri için
block_size uzunluğunda bir girdi penceresi (x) ile onun bir kaydırılmış hali
olan hedefi (y) çıkarır.

Bu kaydırma, kendi kendine denetimli (self-supervised) hedefin ta kendisidir:
ayrıca etiket gerekmez, her tokenın etiketi bir sonraki tokendır.
"""

import torch

def get_batch(data, block_size, batch_size)->tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(0,len(data)-block_size,(batch_size,))

    inputs = []
    for start in ix:
        inputs.append(data[start:start+block_size])

    targets = []
    for start in ix:
        targets.append(data[start+1:start+block_size+1])

    x = torch.stack(inputs)

    y = torch.stack(targets)

    return x, y

# Manual test
# if __name__=='__main__':
#     train_data= torch.load('../../data/processed/train.pt')
#     x, y = get_batch(train_data, 8, 4)
#     print(x.shape, y.shape)
