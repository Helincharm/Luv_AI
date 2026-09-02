"""Eğitim durumunun diske kaydı ve geri yüklenmesi.

Model ağırlıklarını ve optimizer durumunu birlikte saklar; eğitimi kaldığı
yerden sürdürmeyi mümkün kılar.

Optimizer'ın da kaydedilmesi gerekir: Adam her parametre için momentum ve
ikinci moment tahminleri tutar, bunlar atılırsa eğitim kaldığı yerden değil
sarsıntılı bir yeniden ısınmayla devam eder.

Model konfigürasyonu da checkpoint'e gömülür. Ağırlık tensörlerinin şekli
n_embd, n_layer, n_head ve block_size'a bağlı olduğu için, yükleyen taraf
modeli kurmadan önce bu değerleri bilmek zorundadır; koda gömülürse farklı
boyutta eğitilmiş bir checkpoint sessizce uyumsuz hale gelir.

Yükleme sırasında optimizer isteğe bağlıdır -- çıkarımda optimizer yoktur.
map_location ise GPU'da eğitilmiş bir checkpoint'in CPU'da açılmasını sağlar.
"""

import torch


def save_checkpoint(model, optimizer, path, config=None):
    checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
    if config is not None:
        checkpoint["config"] = config
    torch.save(checkpoint, path)


def load_checkpoint(path, model=None, optimizer=None, map_location="cpu"):
    checkpoint = torch.load(path, map_location=map_location)
    if model is not None:
        model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint
