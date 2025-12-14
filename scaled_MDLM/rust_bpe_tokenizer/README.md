# High-Performance BPE Tokenizer (Rust)

A multi-threaded, memory-efficient Byte Pair Encoding tokenizer optimized for processing large datasets like TinyStories (~2.5GB).

## Features

- **Memory-efficient training**: Uses word frequency deduplication and sampling
- **Parallel processing**: Uses `rayon` for multi-threaded pair counting and merging
- **In-place merging**: Avoids allocating new vectors during training
- **Streaming encoding**: Processes in chunks to limit peak memory usage
- **Memory-mapped I/O**: Efficiently handles large files
- **Special token support**: Preserves tokens like `<|endoftext|>` as single tokens
- **Python-compatible output**: Saves tokenizer in same JSON format as the Python version
- **NumPy output**: Saves encoded data as uint16 `.npy` file

## Build

```bash
# Release build (highly optimized)
cargo build --release

# The binary will be at: target/release/bpe_tokenizer
```

## Usage

```bash
# Train on TinyStories (auto-detects <|endoftext|> separator)
./target/release/bpe_tokenizer -i TinyStories-train.txt

# Use smaller sample for less RAM (e.g., 256MB)
./target/release/bpe_tokenizer -i TinyStories-train.txt --sample-mb 256

# Use full text for training (more RAM, slightly better quality)
./target/release/bpe_tokenizer -i TinyStories-train.txt --sample-mb 0

# Custom separator token (or empty to disable)
./target/release/bpe_tokenizer -i data.txt --text-separator "<|end|>"
./target/release/bpe_tokenizer -i data.txt --text-separator ""

# Full options for training
./target/release/bpe_tokenizer \
    -i TinyStories-train.txt \
    -t tokenizer_bpe.json \
    -o encoded_data.npy \
    -v 8191 \
    --sample-mb 512 \
    --threads 0

# ENCODE-ONLY MODE: Use existing tokenizer on a different dataset
./target/release/bpe_tokenizer \
    -i TinyStories-valid.txt \
    -t tokenizer_bpe.json \
    -o valid_encoded.npy \
    --encode-only

# Help
./target/release/bpe_tokenizer --help
```

### Arguments

| Flag | Long | Default | Description |
|------|------|---------|-------------|
| `-i` | `--input` | (required) | Path to input text file |
| `-t` | `--tokenizer` | `tokenizer_bpe.json` | Path to save/load tokenizer JSON |
| `-o` | `--output` | `encoded_data.npy` | Path to save encoded numpy array |
| `-v` | `--vocab-size` | `8191` | Target vocabulary size (before `<MASK>`) |
| `-e` | `--encode-only` | `false` | Load existing tokenizer and encode (skip training) |
| | `--sample-mb` | `512` | Training sample size in MB (0 = full text) |
| | `--text-separator` | `<\|endoftext\|>` | Special token in text to preserve (empty to disable) |
| | `--threads` | `0` | Number of threads (0 = auto) |

## Special Tokens

The tokenizer handles two types of special tokens:

1. **Text separator** (e.g., `<|endoftext|>`): Found in the dataset, preserved as a single token during training. This separates stories/documents.

2. **`<MASK>`**: Added after training for masked language modeling tasks.

Final vocab structure (8192 tokens total):
- IDs 0-242: Single characters (~243)
- ID 243: `<|endoftext|>` (text separator)
- IDs 244-8190: BPE merges (~7947)
- ID 8191: `<MASK>`

## Memory Usage

The `--sample-mb` flag controls peak memory during training:

| Sample Size | Peak RAM (approx) | Quality |
|-------------|-------------------|---------|
| 256 MB | ~2-4 GB | Good |
| 512 MB | ~4-8 GB | Very Good |
| 1024 MB | ~8-12 GB | Excellent |
| 0 (full) | ~15-25 GB | Best |

For a 16GB MacBook, `--sample-mb 512` (default) should work well.

## Output Files

### tokenizer_bpe.json
```json
{
  "vocab": {"0": "a", "1": "b", ..., "243": "<|endoftext|>", ..., "8191": "<MASK>"},
  "merges": [[0, 1], [2, 3], ...],
  "special_tokens": {"<|endoftext|>": 243, "<MASK>": 8191},
  "max_token_len": 15
}
```

### encoded_data.npy
- NumPy array of dtype `uint16`
- Load in Python: `np.load("encoded_data.npy")`
- Or with PyTorch: `torch.from_numpy(np.load("encoded_data.npy")).long()`

## Performance

On an 8-core CPU with 16GB RAM:
- TinyStories (~2GB): ~3-8 minutes training, ~1-2 minutes encoding
- Compression ratio: typically 3-4x (characters to tokens)

## Loading in Python

```python
import json
import numpy as np
import torch

# Load tokenizer
with open('tokenizer_bpe.json') as f:
    data = json.load(f)
    
vocab = {int(k): v for k, v in data['vocab'].items()}
special_tokens = {k: int(v) for k, v in data['special_tokens'].items()}

# Load encoded data
encoded = np.load('encoded_data.npy')
tensor = torch.from_numpy(encoded).long()

print(f"Vocabulary size: {len(vocab)}")
print(f"Encoded tokens: {len(encoded):,}")
print(f"Special tokens: {special_tokens}")

# Find story boundaries
eot_id = special_tokens['<|endoftext|>']
story_ends = (tensor == eot_id).nonzero(as_tuple=True)[0]
print(f"Number of stories: {len(story_ends)}")
```
