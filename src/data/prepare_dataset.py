"""Ham metinden eğitime hazır tensörlere giden veri hazırlama adımları.

Metni okur, train/val olarak ayırır ve token dizisini diske PyTorch tensörü
olarak yazar. Bölme sıraya sadık yapılır (karıştırılmadan), çünkü dil
modelinde komşu tokenlar arasındaki bağlam korunmalıdır.

Modül yalnızca yardımcı fonksiyonlar içerir; adımların sırasını
scripts/tokenize_data.py yönetir.
"""

import torch


def load_raw_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def split_train_val(ids: list[int], val_ratio: float = 0.1) -> tuple[list[int], list[int]]:
    split_idx = int(len(ids) * (1 - val_ratio))
    train = ids[:split_idx]
    val =ids[split_idx:]
    return train, val


def save_encoded(ids: list[int], path: str) -> None:
    # duz python listesiyle degil, PyTorch'un kendi veri yapisi olan
    # TENSOR ile calisiyo. önce listeyi tensore ceviriyom.
    tensor = torch.tensor(ids, dtype=torch.long)
    torch.save(tensor, path)

# Manual test
# if __name__ == "__main__":
#     text = load_raw_text("../../data/raw/sample.txt")
#     print("raw datadaki toplam karakter", len(text))
