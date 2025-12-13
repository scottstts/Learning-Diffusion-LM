import re
from collections import Counter
import json
import os

class SimpleBPE:
    """
    Simple Byte Pair Encoding tokenizer.
    Trains custom vocabulary on your text data.
    """
    
    def __init__(self):
        self.vocab = {}           # token_id -> token_string
        self.vocab_inv = {}       # token_string -> token_id
        self.merges = []          # list of merge rules
        self.special_tokens = {}
        self.max_token_len = 1    # for fast encoding
        
    def _get_stats(self, token_seqs):
        """Count frequency of adjacent pairs."""
        pairs = Counter()
        for seq in token_seqs:
            for i in range(len(seq) - 1):
                pairs[(seq[i], seq[i + 1])] += 1
        return pairs
    
    def _merge_pair(self, token_seqs, pair, new_token):
        """Merge all occurrences of pair into new_token."""
        new_seqs = []
        for seq in token_seqs:
            new_seq = []
            i = 0
            while i < len(seq):
                if i < len(seq) - 1 and seq[i] == pair[0] and seq[i + 1] == pair[1]:
                    new_seq.append(new_token)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1
            new_seqs.append(new_seq)
        return new_seqs
    
    def train(self, text, vocab_size=1000, verbose=True):
        """Train BPE on the given text."""
        if verbose:
            print(f"Training BPE tokenizer on {len(text):,} characters...")
        
        # Split into words (keep whitespace attached)
        words = re.findall(r'\S+|\s+', text)
        token_seqs = [[c for c in word] for word in words]
        
        # Initial vocabulary = unique characters
        chars = sorted(set(text))
        self.vocab = {i: c for i, c in enumerate(chars)}
        self.vocab_inv = {c: i for i, c in enumerate(chars)}
        next_id = len(chars)
        
        if verbose:
            print(f"Base vocabulary: {len(chars)} characters")
        
        # Convert to token IDs
        token_seqs = [[self.vocab_inv[c] for c in seq] for seq in token_seqs]
        
        # Iteratively merge most frequent pairs
        self.merges = []
        num_merges = vocab_size - len(chars)
        
        for i in range(num_merges):
            stats = self._get_stats(token_seqs)
            if not stats:
                break
                
            best_pair = max(stats, key=stats.get)
            new_token_str = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            
            self.vocab[next_id] = new_token_str
            self.vocab_inv[new_token_str] = next_id
            self.merges.append(best_pair)
            
            token_seqs = self._merge_pair(token_seqs, best_pair, next_id)
            
            if verbose and (i + 1) % 200 == 0:
                print(f"  {i+1}/{num_merges} merges completed...")
            
            next_id += 1
        
        # Set max token length for fast encoding
        self.max_token_len = max(len(t) for t in self.vocab.values())
        
        if verbose:
            print(f"Training complete! Vocabulary size: {len(self.vocab)}")
            
        return self
    
    def add_special_token(self, token_str):
        """Add a special token like <MASK>."""
        token_id = len(self.vocab)
        self.vocab[token_id] = token_str
        self.vocab_inv[token_str] = token_id
        self.special_tokens[token_str] = token_id
        self.max_token_len = max(self.max_token_len, len(token_str))
        return token_id
    
    def encode(self, text):
        """
        Fast encoding using greedy longest-match.
        O(max_token_len * text_len) instead of O(num_merges * text_len)
        """
        tokens = []
        i = 0
        n = len(text)
        
        while i < n:
            # Try longest match first, then shorter
            for length in range(min(self.max_token_len, n - i), 0, -1):
                substr = text[i:i + length]
                if substr in self.vocab_inv:
                    tokens.append(self.vocab_inv[substr])
                    i += length
                    break
            else:
                raise ValueError(f"Unknown character at position {i}: {repr(text[i])}")
        
        return tokens
    
    def decode(self, token_ids):
        """Decode token IDs back to text."""
        return ''.join(self.vocab[i] for i in token_ids)
    
    def save(self, path):
        """Save tokenizer to file."""
        with open(path, 'w') as f:
            json.dump({
                'vocab': {str(k): v for k, v in self.vocab.items()},
                'merges': self.merges,
                'special_tokens': self.special_tokens,
                'max_token_len': self.max_token_len
            }, f)
        print(f"Tokenizer saved to {path}")
    
    def load(self, path):
        """Load tokenizer from file."""
        with open(path) as f:
            data = json.load(f)
        self.vocab = {int(k): v for k, v in data['vocab'].items()}
        self.vocab_inv = {v: k for k, v in self.vocab.items()}
        self.merges = [tuple(m) for m in data['merges']]
        self.special_tokens = data.get('special_tokens', {})
        self.max_token_len = data.get('max_token_len', max(len(t) for t in self.vocab.values()))
        print(f"Tokenizer loaded from {path} (vocab size: {len(self.vocab)})")
        return self


if __name__ == "__main__":
    import torch
    
    # Config
    TEXT_PATH = 'harry_potter.txt'
    TOKENIZER_PATH = 'tokenizer_bpe.json'
    TENSOR_PATH = 'encoded_data.pt'
    VOCAB_SIZE = 8191
    
    # Load text
    print(f"Loading text from {TEXT_PATH}...")
    with open(TEXT_PATH, 'r', encoding='utf-8') as f:
        text = f.read()
    print(f"Text length: {len(text):,} characters")
    
    # Train or load tokenizer
    tokenizer = SimpleBPE()
    
    if os.path.exists(TOKENIZER_PATH):
        print(f"\nTokenizer already exists, loading...")
        tokenizer.load(TOKENIZER_PATH)
    else:
        print(f"\nTraining new tokenizer...")
        tokenizer.train(text, vocab_size=VOCAB_SIZE)
        tokenizer.add_special_token('<MASK>')
        tokenizer.save(TOKENIZER_PATH)
    
    print(f"Vocab size: {len(tokenizer.vocab)}")
    print(f"MASK token: {tokenizer.special_tokens.get('<MASK>', 'Not found')}")
    
    # Encode full corpus
    print(f"\nEncoding full corpus...")
    encoded = tokenizer.encode(text)
    print(f"Encoded to {len(encoded):,} tokens")
    print(f"Compression ratio: {len(text) / len(encoded):.2f}x")
    
    # Save as tensor
    tensor = torch.tensor(encoded, dtype=torch.long)
    torch.save(tensor, TENSOR_PATH)
    print(f"Tensor saved to {TENSOR_PATH} ({tensor.numel() * tensor.element_size() / 1e6:.1f} MB)")
    
    # Verify
    print(f"\nVerification:")
    sample = tensor[:100].tolist()
    decoded = tokenizer.decode(sample)
    print(f"First 100 tokens decode to: {decoded[:200]}...")
    
    print("\nDone!")