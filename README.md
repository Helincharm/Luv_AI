# Luv AI

A **decoder-only Transformer language model built from scratch in PyTorch**, focused on Turkish text generation.

No `transformers` library. No pretrained weights.
The tokenizer, attention mechanism, causal masking, Transformer blocks, training loop, checkpointing, and token sampling are implemented directly with PyTorch.

---

## Overview

Luv AI is designed to explore the internal mechanics of modern language models by implementing the complete training pipeline from the ground up.

The project currently includes:

* Character-level tokenization
* Learned token and positional embeddings
* Multi-head self-attention
* Causal masking
* Pre-norm Transformer blocks
* Feed-forward networks
* Cross-entropy language modeling objective
* AdamW optimization
* Reproducible training
* Model and optimizer checkpointing
* Temperature-controlled token sampling

Not yet implemented:

* Autoregressive text generation and the chat interface
* BPE tokenization and a full Turkish corpus
* Validation loss tracking and device management
* Scaling infrastructure — learning rate scheduling, gradient clipping, mixed precision, gradient accumulation
* Unit test coverage

---

## Training Verification

The current small configuration is used to verify that the full pipeline trains correctly end to end.

```text
step   0: loss 3.7816
step 100: loss 1.4456
step 200: loss 0.5093
step 300: loss 0.1970
step 400: loss 0.1355
```

The initial loss is close to the expected `ln(38)` baseline for a 38-symbol vocabulary.

The final loss reflects deliberate memorization of the small test corpus rather than generalization. This serves as a sanity check that gradients, masking, optimization, and checkpointing are functioning correctly before scaling the model.

---

## Architecture

| Configuration      | Small | Medium | Large *(target)* |
| ------------------ | ----: | -----: | ---------------: |
| Context length     |    32 |    256 |              512 |
| Embedding size     |    64 |    384 |              768 |
| Attention heads    |     4 |      6 |               12 |
| Transformer layers |     2 |      6 |               12 |
| Dropout            |   0.0 |    0.2 |              0.1 |
| Parameters         | ~0.1M |   ~11M |            ~100M |

Small and Medium are defined in `config/model_config.py`. Large is the target configuration once the scaling work is in place.

---

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Tokenize the dataset:

```bash
python scripts/tokenize_data.py
```

Train the model:

```bash
python scripts/train.py
```

---

## Project Structure

```text
config/        Model and training configuration
data/          Raw and processed datasets
scripts/       Training and preprocessing entry points
src/
├── data/      Tokenizer and dataset preparation
├── model/     Transformer architecture
├── training/  Training, optimization and checkpointing
├── inference/ Token sampling
└── utils/     Reproducibility
```

Modules for text generation, the chat interface and the test suite are listed
under *Not yet implemented* above and will be added as they are written.

---

Built as a hands-on implementation of Transformer language modeling, with an emphasis on understanding the complete system rather than relying on high-level pretrained model libraries.
