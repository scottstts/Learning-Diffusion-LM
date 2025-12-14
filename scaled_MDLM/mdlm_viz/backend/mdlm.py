import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ============================================================================
# Model Architecture
# ============================================================================

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.unsqueeze(1) * emb.unsqueeze(0) * 1000
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.silu(gate)


class MHA(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.1):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        self.c_attn = nn.Linear(n_embd, 3*n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)

        from torchtune.modules import RotaryPositionalEmbeddings
        self.rope = RotaryPositionalEmbeddings(dim=n_embd // n_head)

        self.attn_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim)
        q = self.rope(q)
        q = q.transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim)
        k = self.rope(k)
        k = k.transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)*(1.0 / math.sqrt(self.head_dim)))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.residual_dropout(self.c_proj(out))
        return out


class FFN(nn.Module):
    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 8*n_embd)
        self.swiglu = SwiGLU()
        self.c_proj = nn.Linear(4*n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.swiglu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.1, use_time=True):
        super().__init__()
        self.use_time = use_time
        self.rms_norm_1 = nn.RMSNorm(n_embd)
        self.rms_norm_2 = nn.RMSNorm(n_embd)
        self.attn = MHA(n_embd, n_head, dropout)
        self.ffn = FFN(n_embd, dropout)
        if use_time:
            self.time_ffn = nn.Sequential(
                nn.Linear(n_embd, 2 * n_embd),
                SwiGLU(),
                nn.Linear(n_embd, 4*n_embd)
            )

    def forward(self, x, time_emb=None):
        if self.use_time and time_emb is not None:
            time_params = self.time_ffn(time_emb)
            shift1, scale1, shift2, scale2 = time_params.chunk(4, dim=-1)
            h = self.rms_norm_1(x) * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
            x = x + self.attn(h)
            h = self.rms_norm_2(x) * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
            x = x + self.ffn(h)
        else:
            x = x + self.attn(self.rms_norm_1(x))
            x = x + self.ffn(self.rms_norm_2(x))
        return x


class MDLM(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_block, block_size, dropout=0.1, use_time=True):
        super().__init__()
        self.block_size = block_size
        self.use_time = use_time
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        if use_time:
            self.time_emb = SinusoidalTimeEmbedding(n_embd)
            self.time_proj = nn.Sequential(
                nn.Linear(n_embd, 2*n_embd),
                SwiGLU(),
                nn.Linear(n_embd, n_embd)
            )
        self.blocks = nn.ModuleList([
            Block(n_embd, n_head, dropout, use_time) for _ in range(n_block)
        ])
        self.rms_norm_final = nn.RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self.dropout = nn.Dropout(dropout)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.RMSNorm):
            torch.nn.init.ones_(module.weight)

    def forward(self, x, t=None):
        B, T = x.shape
        tok_emb = self.tok_emb(x)
        h = self.dropout(tok_emb)
        if self.use_time and t is not None:
            t_emb = self.time_emb(t)
            t_emb = self.time_proj(t_emb)
        else:
            t_emb = None
        for block in self.blocks:
            h = block(h, t_emb)
        h = self.rms_norm_final(h)
        logits = self.lm_head(h)
        return logits


# ============================================================================
# Sampling with State Recording
# ============================================================================

@torch.no_grad()
def sample_with_history(model, seq_len, mask_token, num_steps, temperature, top_p, device):
    """
    Generate text using the reverse diffusion process with ancestral sampling,
    recording all intermediate states.

    Returns:
        history: List of token sequences (one per step)
    """
    model.eval()
    history = []

    # Start with all masks
    x = torch.full((1, seq_len), mask_token, dtype=torch.long, device=device)
    history.append(x[0].tolist())

    # Linearly spaced time steps from 1 to 0
    timesteps = torch.linspace(1, 0, num_steps + 1, device=device)

    for i in range(num_steps):
        t_current = timesteps[i]
        t_next = timesteps[i + 1]

        # Get model predictions
        t_batch = torch.tensor([t_current], device=device)
        logits = model(x, t_batch)

        if top_p < 1.0:
            # Sort logits in descending order
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            
            # Calculate cumulative probabilities
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above top_p
            sorted_indices_to_remove = cumulative_probs > top_p
            
            # handle edge case where top_p is too small
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # 4. Scatter mask back to original indices
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            
            # 5. Set filtered logits to -inf
            logits[indices_to_remove] = float('-inf')

        # Don't predict [MASK] token during sampling
        logits[:, :, mask_token] = float('-inf')

        # Apply temperature
        logits = logits / temperature

        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)

        # Sample predictions for all positions
        pred_tokens = torch.multinomial(
            probs.view(-1, probs.size(-1)),
            num_samples=1
        ).view(1, seq_len)

        # Ancestral sampling: roll a probability die for each token
        # Calculate the probability of unmasking at this step
        if t_current > 0:
            p_unmask = (t_current - t_next) / t_current
        else:
            p_unmask = 1.0  # Force finish at the very end

        # Roll the dice for every position
        random_values = torch.rand_like(probs[:, :, 0])  # Shape: [1, seq_len]
        should_unmask = random_values < p_unmask

        # Only update if it is CURRENTLY a mask AND the dice roll said "Unmask"
        is_mask = (x == mask_token)
        update_mask = is_mask & should_unmask

        x[update_mask] = pred_tokens[update_mask]

        history.append(x[0].tolist())

    # Final cleanup: unmask any remaining masks
    is_mask = (x == mask_token)
    if is_mask.any():
        logits = model(x, torch.tensor([0.0], device=device))
        logits[:, :, mask_token] = float('-inf')
        probs = F.softmax(logits / temperature, dim=-1)
        final_preds = probs.argmax(dim=-1)
        x[is_mask] = final_preds[is_mask]
        history.append(x[0].tolist())

    return history
