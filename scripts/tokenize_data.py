"""Veri hazırlama boru hattının giriş noktası.

src/data içindeki parçaları doğru sırada çağırır: ham metni okur,
tokenizer'ı kurar, encode eder, train/val olarak böler ve sonuçları
data/processed altına yazar (train.pt, val.pt, vocab.json).

Kendi mantığı yoktur, yalnızca orkestrasyon yapar. Sözlük ayrıca vocab.json
olarak saklanır; çıkarım tarafının eğitimdekiyle birebir aynı eşlemeyi
kullanması zorunludur, aksi halde aynı sayı farklı harfe çözülür.

    python scripts/tokenize_data.py
"""

import json
import os
import sys

# scripts/ icinden calistirildiginda src/ altindaki paketleri bulabilmek icin
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from data.tokenizer import CharTokenizer
from data.prepare_dataset import load_raw_text, split_train_val, save_encoded

RAW_PATH = "data/raw/sample.txt"
PROCESSED_DIR = "data/processed"

os.makedirs(PROCESSED_DIR, exist_ok=True)
text = load_raw_text(RAW_PATH)
tokenizer = CharTokenizer(text)
ids = tokenizer.encode(text)
train_ids, val_ids = split_train_val(ids, val_ratio=0.1)

save_encoded(train_ids, f"{PROCESSED_DIR}/train.pt")
save_encoded(val_ids, f"{PROCESSED_DIR}/val.pt")

vocab = {"stoi": tokenizer.stoi, "itos": tokenizer.itos}
with open(f"{PROCESSED_DIR}/vocab.json", "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False, indent=2)

print("vocab_size:", tokenizer.vocab_size)
print("train uzunluk:", len(train_ids))
print("val uzunluk:", len(val_ids))
print("oran (val/toplam):", len(val_ids) / len(ids))
print("decode(encode(metin)) == metin mi?", tokenizer.decode(ids) == text)
