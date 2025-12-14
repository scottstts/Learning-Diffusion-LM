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
    CKPT_PATH = os.path.join(BASE_DIR, 'latest.pt')
    
    BLOCK_SIZE = 1024
    N_EMBD = 768
    N_HEAD = 12
    N_BLOCKS = 12
    
    SEQ_LEN = 1024
    NUM_STEPS = 50 # Default, can be overridden
    TOP_P = 0.9

# ============================================================================
# Minimal Tokenizer (to replace bpe_tokenizer.SimpleBPE)
# ============================================================================
class SimpleBPE:
    def __init__(self):
        self.vocab = {}
        self.special_tokens = {}
        self.inv_vocab = {}

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
            
            # Identify special tokens
            # We need to find the ID for <MASK> (which is a value in raw_vocab, or usually we look up by string)
            # Create inverse vocab first for easy lookup
            self.inv_vocab = {v: k for k, v in self.vocab.items()}
            
            if '<MASK>' in self.inv_vocab:
                self.special_tokens['<MASK>'] = self.inv_vocab['<MASK>']
            if '<|endoftext|>' in self.inv_vocab:
                self.special_tokens['<|endoftext|>'] = self.inv_vocab['<|endoftext|>']
                
        return self
def main():
    # 1. Parse Arguments (JSON from stdin or args)
    # Expected input: { "temperature": 1.0, "steps": 50 }
    try:
        if len(sys.argv) > 1:
            # Maybe passed as string arg?
            args = json.loads(sys.argv[1])
        elif not sys.stdin.isatty():
             # Read from stdin
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

    temperature = float(args.get("temperature", 1.0))
    steps = int(args.get("steps", Config.NUM_STEPS))
    
    # 2. Setup Device
    device = torch.device("cpu") # Force CPU for local web server friendliness, or check CUDA
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    
    # 3. Load Tokenizer
    tokenizer = SimpleBPE().load(Config.TOKENIZER_PATH)
    mask_token = tokenizer.special_tokens.get('<MASK>')
    if mask_token is None:
        # Fallback if not found in vocab
        mask_token = 0 
        sys.stderr.write("Warning: <MASK> token not found in vocab, using 0\n")

    vocab_size = len(tokenizer.vocab)

    # 4. Load Model
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
            
            # Clean state dict keys if needed
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
            # Continue with random weights? Or failed.
            # Ideally fail, but for demo let's continue.
    else:
        sys.stderr.write(f"Warning: Checkpoint not found at {Config.CKPT_PATH}. Using random weights.\n")

    # 5. Run Inference
    history = sample_with_history(
        model,
        seq_len=Config.SEQ_LEN,
        mask_token=mask_token,
        num_steps=steps,
        temperature=temperature,
        top_p=Config.TOP_P,
        device=device
    )

    # 6. Output JSON result
    # We want to send back:
    # - history: list of [ [token_id, ...], ... ]
    # - vocab: map of id -> string (so frontend can render)
    #   Sending full vocab might be heavy. 
    #   Alternatively, we can just send the string representation of history?
    #   Or send the used tokens' strings.
    #   For "visualization", knowing the specific tokens is useful.
    #   Let's send tokens + a subset of vocab or dynamic lookup?
    #   The existing pygame visualizer decodes on the fly.
    #   Let's send valid strings for the tokens used in the history.
    
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
