# Luv AI

**English** · [Türkçe](README.tr.md)

A GPT-style Transformer language model written from scratch in PyTorch. No model
libraries: every part — embedding layers, self-attention, the transformer block,
the training loop, sampling — is implemented by hand.

End goal: a model that generates Turkish text and can be talked to over the web.

## Status

Phase 1 (core architecture and training pipeline) is complete and runs end to
end: the model trains and produces a loadable checkpoint. Text generation and
the chat interface are next.

```
step   0: loss 3.7816
step 100: loss 1.4456
step 200: loss 0.5093
step 300: loss 0.1970
step 400: loss 0.1355
```

The starting loss matches `ln(vocab_size)`, which is what an untrained model
should score. The current sample corpus is tiny, so the low final loss is
memorisation — expected at this stage, and enough to verify the pipeline.

## Roadmap

### Phase 1 — Core architecture and training pipeline

- [x] Character-level tokenizer with lossless encode/decode round-trip
- [x] Data preparation from raw text to train/val tensors
- [x] Token and positional embedding layers
- [x] Causally masked self-attention (single and multi-head)
- [x] Pre-norm transformer block with residual connections
- [x] End-to-end GPT model
- [x] Cross-entropy loss and backpropagation
- [x] Random-window batch sampling
- [x] Training loop and runnable entry point
- [x] Checkpoint saving and loading
- [ ] Autoregressive text generation *(`sampler.py` done, `generate.py` pending)*
- [ ] Terminal chat interface

### Phase 2 — Hardening

- [ ] Restore integer keys for `itos` when loading `vocab.json`
- [ ] Embed model configuration into the checkpoint
- [ ] `map_location` support in `torch.load` (train on GPU, load on CPU)
- [ ] Optimizer-free checkpoint loading for inference
- [ ] Validation loss measurement
- [ ] `model.train()` / `model.eval()` mode handling
- [ ] Device (CPU/GPU) management
- [ ] Hyperparameter validation and edge-case guards

### Phase 3 — Scaling infrastructure

Prerequisites for training a larger model. Without these, training either
diverges or runs out of memory.

- [ ] Learning-rate schedule (warmup + cosine decay)
- [ ] Gradient clipping (`clip_grad_norm_`)
- [ ] Mixed-precision training (AMP / bf16)
- [ ] Gradient accumulation for large effective batches on small GPUs
- [ ] Memory-mapped data loading (`np.memmap`)
- [ ] Resume training from a checkpoint, step counter included
- [ ] Selective weight decay (excluding biases and LayerNorm parameters)
- [ ] Keep the checkpoint with the best validation loss

### Phase 4 — Data and tokenizer

- [ ] Collect and clean a Turkish corpus
- [ ] BPE (subword) tokenizer — `src/data/bpe_tokenizer.py`
- [ ] Tokenizer training script — `scripts/train_tokenizer.py`
- [ ] Tokenizer selection in `tokenize_data.py`
- [ ] Full-scale training run on GPU

### Phase 5 — Model quality and speed

- [ ] Weight tying between the embedding and the LM head
- [ ] Wire dropout into the model (defined in config, unused so far)
- [ ] GELU instead of ReLU
- [ ] Scaled weight initialisation (normal std=0.02, depth-scaled residuals)
- [ ] Fused QKV projection instead of a per-head loop
- [ ] `F.scaled_dot_product_attention` (FlashAttention)
- [ ] KV cache during generation
- [ ] `torch.compile`
- [ ] Rotary position embeddings (RoPE)

### Phase 6 — Evaluation

- [ ] Perplexity metric
- [ ] Periodic sample generation during training
- [ ] Separate test split (train / val / test)
- [ ] Reproducible evaluation over fixed validation batches
- [ ] Throughput measurement (tokens/sec, MFU)

### Phase 7 — Conversational behaviour

- [ ] Chat-formatted dataset (user / assistant structure)
- [ ] Fine-tuning pass
- [ ] Stop-sequence handling during generation
- [ ] System prompt support

### Phase 8 — Web layer

- [ ] FastAPI server with a `/chat` endpoint
- [ ] Web interface
- [ ] Token streaming over SSE
- [ ] Conversation history management and context trimming
- [ ] Request batching
- [ ] Rate limiting, timeouts, health-check endpoint
- [ ] Deployment

### Phase 9 — Engineering maturity

- [ ] Package with `pyproject.toml` (removes the `sys.path.insert` shim)
- [ ] `argparse` command-line arguments for the scripts
- [ ] Read configuration from YAML/JSON, override from the CLI
- [ ] Experiment tracking (TensorBoard or Weights & Biases)
- [ ] Write training logs to `reports/`
- [ ] `ruff` + `black` with a pre-commit hook
- [ ] pytest coverage across all modules
- [ ] CI via GitHub Actions
- [x] Pinned dependency versions
- [ ] Model card — parameter count, data, loss and sample outputs

## Highest-Impact Next Steps

Once Phase 1 is complete, in order:

1. **Mixed precision (AMP)** — two to three times faster, half the memory
2. **LR schedule + gradient clipping** — required for stable training at scale
3. **KV cache** — 10-50× faster generation, makes chat usable
4. **Weight tying** — free parameter savings and a quality gain
5. **Periodic sample generation** — shows what works before the loss curve does

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Usage

```bash
# Tokenize the raw text; produces train/val tensors and vocab.json
python scripts/tokenize_data.py
```

```bash
# Train the model; writes checkpoints/luv_ai.pt
python scripts/train.py
```

<!-- Chat usage will be added here once the interface is finished. -->

## Project Structure

```
config/     Model and training hyperparameters
data/       raw/ source text, processed/ tokenized tensors
scripts/    Runnable entry points
src/
    data/       Tokenizer and data preparation
    model/      Embeddings, attention, transformer block, GPT
    training/   Batching, loss, optimizer, checkpointing, training loop
    inference/  Sampling and text generation
    cli/        Terminal interface
    utils/      Seeding and logging
tests/      Unit tests
```

## Architecture Notes

A decoder-only, GPT-style transformer:

- Learned token and positional embeddings
- Causally masked multi-head self-attention
- Pre-norm layout with residual connections
- `n_embd -> 4*n_embd -> n_embd` feed-forward layer

### Configurations

| | `small` | `medium` | `large` *(planned)* |
|---|---|---|---|
| block_size | 32 | 256 | 512 |
| n_embd | 64 | 384 | 768 |
| n_head | 4 | 6 | 12 |
| n_layer | 2 | 6 | 12 |
| dropout | 0.0 | 0.2 | 0.1 |
| ~parameters | 0.1 M | 11 M | ~100 M |

`small` verifies the pipeline quickly, `medium` is the first real training run,
and `large` is the target. `large` sits in the same size class as GPT-2 small
and is trainable on a single 16 GB GPU with mixed precision and gradient
accumulation — which is what Phase 3 exists to make possible.

The tokenizer is character-level for now; Phase 4 moves it to BPE.

## Notes

This is a learning project built step by step, and it is still in progress.
Unchecked items above are genuinely unimplemented rather than aspirational
filler — the checkboxes are the honest state of the repository.
