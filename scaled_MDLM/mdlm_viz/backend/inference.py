import os
import sys
import json
import torch
import torch.nn.functional as F
from mdlm import MDLM, sample_with_history

# ============================================================================
# Configuration (Mirrors ref/sample_visual_config.py)
# ============================================================================
class Config:
    # Paths (relative to the script itself, to be robust)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    TOKENIZER_PATH = os.path.join(BASE_DIR, 'tokenizer_bpe.json')
    CKPT_PATH = os.path.join(BASE_DIR, 'model_state_fp32.pt')
    
    BLOCK_SIZE = 1024
    N_EMBD = 768
    N_HEAD = 12
    N_BLOCKS = 12
    
    SEQ_LEN = 1024
    NUM_STEPS = 100 # Default, can be overridden
    TOP_P = 0.9

# ============================================================================
# Minimal Tokenizer (to replace bpe_tokenizer.SimpleBPE)
# ============================================================================
class SimpleBPE:
    def __init__(self):
        self.vocab = {}
        self.special_tokens = {}
        self.inv_vocab = {}
        self.merges = []
        self.merge_ranks = {}

    def load(self, path):
        if not os.path.exists(path):
            # Fallback/Mock if file is missing (development safety)
            sys.stderr.write(f"Warning: Tokenizer file not found at {path}. Using mock tokenizer.\n")
            self.vocab = {i: f"token_{i}" for i in range(1000)}
            self.vocab[0] = '<MASK>'
            self.vocab[1] = '<|endoftext|>'
            self.special_tokens = {'<MASK>': 0, '<|endoftext|>': 1}
        else:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                raw_vocab = {}
                # Handle different potential formats of tokenizer.json
                if 'model' in data and 'vocab' in data['model']:
                    raw_vocab = data['model']['vocab']
                elif 'vocab' in data:
                    raw_vocab = data['vocab']
                else:
                    # Assume flat dict
                    raw_vocab = data

                # Convert keys to integers (JSON keys are always strings)
                self.vocab = {int(k): v for k, v in raw_vocab.items()}

                # Load merges if available
                if 'merges' in data:
                    self.merges = data['merges']
                    # Build merge ranks for BPE encoding
                    for i, merge in enumerate(self.merges):
                        self.merge_ranks[tuple(merge)] = i

                # Load special tokens
                if 'special_tokens' in data:
                    self.special_tokens = data['special_tokens']

            # Create inverse vocab for easy lookup
            self.inv_vocab = {v: k for k, v in self.vocab.items()}

            # Fallback special token detection if not in file
            if '<MASK>' not in self.special_tokens and '<MASK>' in self.inv_vocab:
                self.special_tokens['<MASK>'] = self.inv_vocab['<MASK>']
            if '<|endoftext|>' not in self.special_tokens and '<|endoftext|>' in self.inv_vocab:
                self.special_tokens['<|endoftext|>'] = self.inv_vocab['<|endoftext|>']

        return self

    def encode(self, text):
        """Encode text to token IDs using BPE."""
        if not text:
            return []

        # Convert text to initial character-level tokens
        # GPT-2 style: space becomes 'Ġ' character
        tokens = []
        for char in text:
            if char == ' ':
                tokens.append('Ġ')
            else:
                tokens.append(char)

        # Convert to token IDs (character level first)
        token_ids = []
        for tok in tokens:
            if tok in self.inv_vocab:
                token_ids.append(self.inv_vocab[tok])
            else:
                # Unknown character - skip
                pass

        # Apply BPE merges iteratively
        while len(token_ids) >= 2:
            # Find the best merge (lowest rank)
            best_merge = None
            best_rank = float('inf')

            for i in range(len(token_ids) - 1):
                pair = (token_ids[i], token_ids[i + 1])
                if pair in self.merge_ranks:
                    rank = self.merge_ranks[pair]
                    if rank < best_rank:
                        best_rank = rank
                        best_merge = pair

            if best_merge is None:
                break

            # Find the merged token
            tok1 = self.vocab.get(best_merge[0], '')
            tok2 = self.vocab.get(best_merge[1], '')
            merged_str = tok1 + tok2

            if merged_str in self.inv_vocab:
                merged_id = self.inv_vocab[merged_str]
                # Apply merge at all occurrences
                new_token_ids = []
                i = 0
                while i < len(token_ids):
                    if i < len(token_ids) - 1 and (token_ids[i], token_ids[i + 1]) == best_merge:
                        new_token_ids.append(merged_id)
                        i += 2
                    else:
                        new_token_ids.append(token_ids[i])
                        i += 1
                token_ids = new_token_ids
            else:
                break

        return token_ids
def main():
    # 1. Parse Arguments (JSON from stdin or args)
    # Expected input: { "mode": "generate"|"tokenize"|"info", "temperature": 1.0, "steps": 50, "prompt": [...], "text": "..." }
    try:
        if len(sys.argv) > 1:
            args = json.loads(sys.argv[1])
        elif not sys.stdin.isatty():
            input_str = sys.stdin.read()
            if input_str.strip():
                args = json.loads(input_str)
            else:
                args = {}
        else:
            args = {}
    except Exception as e:
        sys.stderr.write(f"Error parsing input: {e}\n")
        args = {}

    mode = args.get("mode", "generate")

    # Load Tokenizer (needed for all modes)
    tokenizer = SimpleBPE().load(Config.TOKENIZER_PATH)
    mask_token = tokenizer.special_tokens.get('<MASK>')
    endoftext_token = tokenizer.special_tokens.get('<|endoftext|>')
    if mask_token is None:
        mask_token = 0
        sys.stderr.write("Warning: <MASK> token not found in vocab, using 0\n")

    # Handle tokenize mode
    if mode == "tokenize":
        text = args.get("text", "")

        # Tokenize word-by-word to track word boundaries
        words = text.split(' ')
        all_token_ids = []
        all_tokens_info = []
        word_boundaries = [0]  # First word always starts at 0

        for i, word in enumerate(words):
            if not word:  # Skip empty strings from multiple spaces
                continue

            # Add space prefix for words after the first
            word_to_tokenize = word if i == 0 else ' ' + word
            word_token_ids = tokenizer.encode(word_to_tokenize)

            # Track where this word starts in token sequence
            if i > 0 and word_token_ids:
                word_boundaries.append(len(all_token_ids))

            all_token_ids.extend(word_token_ids)
            for tid in word_token_ids:
                all_tokens_info.append({
                    "id": tid,
                    "text": tokenizer.vocab.get(tid, '?')
                })

        output = {
            "tokens": all_tokens_info,
            "token_ids": all_token_ids,
            "word_boundaries": word_boundaries
        }
        print(json.dumps(output))
        return

    # Handle info mode (return config info)
    if mode == "info":
        output = {
            "seq_len": Config.SEQ_LEN,
            "mask_token": mask_token,
            "endoftext_token": endoftext_token,
            "vocab_size": len(tokenizer.vocab)
        }
        print(json.dumps(output))
        return

    # Generate mode
    temperature = float(args.get("temperature", 1.0))
    steps = int(args.get("steps", Config.NUM_STEPS))
    prompt = args.get("prompt", None)  # Array of token IDs or None

    # Setup Device
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    vocab_size = len(tokenizer.vocab)

    # Load Model
    model = MDLM(
        vocab_size=vocab_size,
        n_embd=Config.N_EMBD,
        n_head=Config.N_HEAD,
        n_block=Config.N_BLOCKS,
        block_size=Config.BLOCK_SIZE,
    ).to(device)

    if os.path.exists(Config.CKPT_PATH):
        try:
            checkpoint = torch.load(Config.CKPT_PATH, map_location=device, weights_only=False)
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('_orig_mod.'):
                    new_state_dict[key[10:]] = value
                else:
                    new_state_dict[key] = value

            model.load_state_dict(new_state_dict)
            sys.stderr.write(f"Model loaded from {Config.CKPT_PATH}\n")
        except Exception as e:
            sys.stderr.write(f"Failed to load checkpoint: {e}\n")
    else:
        sys.stderr.write(f"Warning: Checkpoint not found at {Config.CKPT_PATH}. Using random weights.\n")

    # Run Inference
    history = sample_with_history(
        model,
        seq_len=Config.SEQ_LEN,
        mask_token=mask_token,
        num_steps=steps,
        temperature=temperature,
        top_p=Config.TOP_P,
        device=device,
        prompt=prompt
    )

    # Output JSON result
    used_token_ids = set()
    for step in history:
        used_token_ids.update(step)

    vocab_subset = {id: tokenizer.vocab.get(id, '?') for id in used_token_ids}

    output = {
        "history": history,
        "vocab": vocab_subset,
        "mask_token": mask_token
    }

    print(json.dumps(output))

if __name__ == "__main__":
    main()
