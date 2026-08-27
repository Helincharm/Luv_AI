"""Eğitimin giriş noktası.

Konfigürasyonu, hazırlanmış veriyi, modeli ve eğitim döngüsünü bir araya
getirip eğitimi başlatır; sonuçta yeniden yüklenebilir bir checkpoint üretir.

Kendi mantığı yoktur, yalnızca orkestrasyon yapar: hiperparametreler
config/model_config.py'den, veri data/processed/ altından, model ve döngü
src/ içinden gelir.

    python scripts/train.py
"""

import json
import os
import sys
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from config.model_config import ModelConfig, TrainConfig  # noqa: E402
from model.gpt import GPT  # noqa: E402
from training.checkpoint import save_checkpoint  # noqa: E402
from training.optimizer import create_optimizer  # noqa: E402
from training.trainer import train  # noqa: E402
from utils.seed import set_seed  # noqa: E402

VOCAB_PATH = os.path.join(ROOT, "data", "processed", "vocab.json")
TRAIN_PATH = os.path.join(ROOT, "data", "processed", "train.pt")
CHECKPOINT_DIR = os.path.join(ROOT, "checkpoints")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "luv_ai.pt")

# print(TRAIN_PATH)

def main() -> None:
    set_seed()

    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    vocab_size = len(vocab["stoi"])
    model_cfg = ModelConfig.small(vocab_size)
    train_cfg = TrainConfig()

    train_data = torch.load(TRAIN_PATH)

    model = GPT(model_cfg.vocab_size,
                model_cfg.block_size,
                model_cfg.n_embd,
                model_cfg.n_head,
                model_cfg.n_layer,
                )

    optimizer = create_optimizer(model, train_cfg.learning_rate)

    train(
        model,
        optimizer,
        train_data,
        model_cfg.block_size,
        train_cfg.batch_size,
        train_cfg.steps,
    )

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    save_checkpoint(model, optimizer, CHECKPOINT_PATH)

if __name__ == "__main__":
    main()
