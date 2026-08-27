# Luv AI

**English** · [Türkçe](README.tr.md)

A decoder-only Transformer language model written from scratch in PyTorch.

No `transformers` library. No pretrained weights. Every component — the
tokenizer, the embedding tables, scaled dot-product attention, the causal mask,
the transformer block, the loss, the optimizer setup, the training loop and the
checkpoint format — is implemented directly on PyTorch tensors and `nn.Module`.

The goal is a model that generates Turkish text and can be served over the web.
This repository is the path to that: the architecture and the training pipeline
are finished and verified end to end; scaling, tokenization and serving are the
work ahead.

---

## Current State

The training pipeline runs end to end and produces a loadable checkpoint.

```
step   0: loss 3.7816
step 100: loss 1.4456
step 200: loss 0.5093
step 300: loss 0.1970
step 400: loss 0.1355
```

Two things are worth reading in these numbers.

The starting loss of **3.78** is close to `ln(38)` — the entropy of a uniform
guess over a 38-symbol vocabulary. A model whose weights are correctly
initialised and whose loss is correctly computed *must* start there. Starting
much higher, or at `nan`, would point to a bug in initialisation, masking or the
loss reduction.

The steady descent to **0.14** is memorisation, not generalisation: the sample
corpus shipped with the repository is a few hundred tokens. That is the point at
this stage. A pipeline that can memorise a tiny corpus is a pipeline whose
gradients flow correctly through every layer — which is exactly what needed
verifying before scaling anything up.

---

## What Is Implemented

### Tokenization and data preparation

A character-level tokenizer builds its vocabulary from the training corpus:
unique characters are sorted and assigned stable indices, with `stoi` and `itos`
as exact inverses. The encode/decode round-trip is lossless and is checked when
the dataset is built.

The corpus is split into training and validation sets **sequentially rather than
randomly**. Shuffling would break the adjacency that a language model learns
from — a token's context must remain the text that actually preceded it.

Encoded splits are written to disk as PyTorch tensors, and the vocabulary is
persisted separately as JSON. The vocabulary file matters more than it looks:
inference must reconstruct the *exact* mapping used during training, or the same
integer decodes to a different character.

### Embeddings

Token identities pass through a learned embedding table. A second learned table
maps each position in the sequence to a vector of the same width, and the two are
summed.

Positional information has to be injected explicitly. Self-attention is
permutation-equivariant — it sees an unordered set — so without a positional
signal the model cannot distinguish "cat sat" from "sat cat".

### Self-attention

Each head projects the input into query, key and value spaces with bias-free
linear layers. Attention scores are the scaled dot product of queries and keys,
divided by `sqrt(head_size)`.

That scaling is not cosmetic. Dot products grow with dimensionality; without the
correction, softmax saturates at larger widths, its gradient collapses toward
zero, and the layer stops learning.

Causality is enforced with a lower-triangular mask registered as a buffer, so it
moves with the model across devices and is excluded from the parameter list.
Positions above the diagonal are set to `-inf` before the softmax, which drives
their weights to exactly zero. This is the constraint that makes next-token
prediction a real prediction rather than a lookup.

Multi-head attention splits the embedding width evenly across heads, runs them
in parallel, concatenates the outputs and passes the result through an output
projection. Divisibility of width by head count is validated at configuration
time rather than surfacing later as a shape-mismatch error deep in a matmul.

Each head can optionally retain its post-softmax weights for inspection. The
flag is off during training so it costs nothing.

### Transformer block

Attention and a position-wise feed-forward network (`n_embd → 4·n_embd →
n_embd`), each wrapped in a residual connection with **pre-norm** placement:
LayerNorm is applied to the input of each sub-layer, not to its output.

Pre-norm is the choice that makes depth practical. Keeping normalisation off the
residual path leaves an unobstructed route for gradients through the whole stack,
which is why deep pre-norm models train stably without the learning-rate warmup
tricks post-norm architectures require.

Attention moves information *between* tokens; the feed-forward layer transforms
it *within* each token. Both are needed, and they do different jobs.

### Model

Embeddings, a stack of transformer blocks, a final LayerNorm, and a linear
language-modelling head producing logits of shape `(batch, sequence,
vocab_size)` — a distribution over the next token at every position.

Training uses all of those positions at once, which is what makes a single
forward pass worth `sequence_length` supervised examples. Generation uses only
the last.

### Training

Batches are cut from the token stream at random offsets: inputs of length
`block_size`, targets the same window shifted by one. That shift *is* the
self-supervised objective — no labels are needed, because every token is the
label for the one before it.

The loss is cross-entropy over the batch and time axes flattened together, so
each position is scored as an independent classification. Softmax is not applied
beforehand; `F.cross_entropy` fuses log-softmax internally for numerical
stability.

Optimisation uses AdamW, which decouples weight decay from the gradient update
and applies regularisation independently of the learning rate — the de facto
standard for transformer training.

Checkpoints store model and optimizer state together. The optimizer half is not
optional: Adam carries per-parameter momentum and second-moment estimates, and
discarding them turns "resume" into a jolting restart.

### Reproducibility and configuration

`random`, `numpy` and `torch` maintain independent generators, so all three are
seeded from one call, with CUDA covered when available. Without a fixed seed
there is no way to tell whether a change or plain luck moved the numbers.

Hyperparameters live in dataclasses, split deliberately in two: architecture and
training settings are separate concerns, and inference needs only the first.
Named presets replace scattered magic numbers, and configuration validity is
checked at construction time with a clear error message rather than at the point
of failure.

---

## Architecture

| | `small` | `medium` | `large` *(target)* |
|---|---|---|---|
| block_size | 32 | 256 | 512 |
| n_embd | 64 | 384 | 768 |
| n_head | 4 | 6 | 12 |
| n_layer | 2 | 6 | 12 |
| dropout | 0.0 | 0.2 | 0.1 |
| parameters | ~0.1 M | ~11 M | ~100 M |

`small` exists to verify the pipeline in seconds. `medium` is the first
configuration where training on real data is meaningful. `large` sits in the same
size class as GPT-2 small and is reachable on a single 16 GB GPU with mixed
precision and gradient accumulation — which is precisely what the scaling work
below is for.

---

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Usage

```bash
# Tokenize the raw corpus; writes train/val tensors and the vocabulary
python scripts/tokenize_data.py
```

```bash
# Train; writes checkpoints/luv_ai.pt
python scripts/train.py
```

## Project Structure

```
config/     Architecture and training hyperparameters
data/       raw/ source text, processed/ tokenized tensors and vocabulary
scripts/    Runnable entry points
src/
    data/       Tokenizer and dataset preparation
    model/      Embeddings, attention, transformer block, GPT
    training/   Batching, loss, optimizer, checkpointing, training loop
    inference/  Sampling and text generation
    cli/        Terminal interface
    utils/      Seeding and logging
tests/      Unit tests
```

---

## What Comes Next

### Finishing the loop

Sampling from logits with temperature control is implemented. What remains is
autoregressive generation — feeding each sampled token back as input, trimming
context to `block_size`, running under `no_grad` — and a terminal interface on
top of it. That closes the path from raw text to a model that talks back.

### Hardening

Several details separate a pipeline that runs from one that survives contact
with a second machine. Model configuration needs to travel *inside* the
checkpoint, so a server can reconstruct the architecture instead of guessing at
it. Loading needs `map_location`, or a model trained on GPU cannot be opened on a
CPU host. Inference needs to load weights without requiring an optimizer.
Validation loss needs to be measured on a schedule — without it there is no
signal for when learning turns into overfitting — and training and evaluation
modes need to be switched explicitly, which becomes load-bearing the moment
dropout is wired in.

### Scaling

Training a larger model is gated on infrastructure, not on architecture. A
warmup-plus-cosine learning-rate schedule and gradient clipping are what keep
deep transformers from diverging in the first few hundred steps. Mixed-precision
training roughly halves memory and multiplies throughput. Gradient accumulation
buys a large effective batch on modest hardware. Memory-mapped loading is what
makes a corpus larger than RAM trainable at all. Resumable training with a
persisted step counter turns a multi-day run into something that survives an
interruption.

### Data and tokenization

Character-level tokenization is the right place to start and the wrong place to
stay. Turkish is agglutinative, and a subword vocabulary learned with BPE lets
the model reuse suffixes as units instead of re-deriving them character by
character — several times more text within the same context window. Collecting
and cleaning a Turkish corpus is the largest single task in the project, and the
one that most determines the final quality.

### Quality and speed

Tying the embedding and output projection saves parameters and typically
improves results. GELU replaces ReLU. Depth-scaled initialisation matters more
as layers stack up. On the performance side: a fused QKV projection replaces the
per-head loop, `F.scaled_dot_product_attention` brings memory-efficient kernels,
`torch.compile` removes interpreter overhead, and a KV cache turns generation
from quadratic re-computation into an incremental one — the difference between a
chat interface that is usable and one that is not.

### Evaluation, conversation and serving

Perplexity, a held-out test split, and periodic sample generation during
training, so quality is observable rather than inferred from a loss curve.
Then conversational behaviour, which is a data problem before it is a code
problem: a base model continues text, it does not answer, and turning one into
the other requires chat-formatted data, a fine-tuning pass and stop-sequence
handling. Finally the serving layer — a FastAPI endpoint with token streaming,
context management and the operational basics.

---

## Notes

This is a learning project, built one component at a time, and it is openly
unfinished. Sections describing future work are descriptions of intent, not of
existing code; everything under *What Is Implemented* is in the repository and
runs.
