"""Eğitim durumunun diske kaydı ve geri yüklenmesi.

Model ağırlıklarını ve optimizer durumunu birlikte saklar; eğitimi kaldığı
yerden sürdürmeyi mümkün kılar.

Optimizer'ın da kaydedilmesi gerekir: Adam her parametre için momentum ve
ikinci moment tahminleri tutar, bunlar atılırsa eğitim kaldığı yerden değil
sarsıntılı bir yeniden ısınmayla devam eder.
"""

import torch

def save_checkpoint(model, optimizer, path):
    checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
    torch.save(checkpoint, path)

def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
