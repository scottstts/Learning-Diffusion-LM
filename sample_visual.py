"""
MDLM Sampling Visualization with Pygame

Visualizes the iterative denoising (unmasking) process from fully masked to fully unmasked text.
Each frame shows the current state of the sequence, with masked tokens displayed as a special symbol.
Uses BPE tokenizer with dynamic cell sizing for variable-length tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import pygame
from bpe_tokenizer import SimpleBPE

# ============================================================================
# Model Architecture (copied from sampling.ipynb)
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
def sample_with_history(model, seq_len, mask_token, num_steps, temperature, device):
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


# ============================================================================
# Pygame Visualization
# ============================================================================

class MDLMVisualizer:
    # Colors - Refined palette
    BG_COLOR = (15, 15, 20)
    GRID_COLOR = (35, 38, 48)
    MASK_BG_COLOR = (45, 35, 55)  # Subtle purple for masked cells
    MASK_COLOR = (180, 130, 200)  # Softer purple-pink for mask symbol
    TEXT_COLOR = (220, 220, 230)  # Soft white for text
    NEWLY_UNMASKED_COLOR = (100, 220, 150)  # Soft green for newly revealed
    NEWLY_UNMASKED_BG = (30, 60, 45)  # Dark green background
    HEADER_COLOR = (130, 200, 235)  # Soft cyan for header
    BUTTON_COLOR = (70, 85, 130)  # Muted blue-purple for button
    BUTTON_HOVER_COLOR = (100, 180, 220)  # Brighter cyan on hover
    BUTTON_TEXT_COLOR = (240, 240, 245)
    PROGRESS_BAR_BG = (30, 32, 40)
    PROGRESS_BAR_FG = (100, 180, 220)

    def __init__(self, model, tokenizer, mask_token, seq_len, device,
                 char_width=14, cell_height=32, fps=10, target_width=1400):
        """
        Initialize the visualizer with dynamic cell sizing for BPE tokens.
        Window height is automatically calculated to fit all tokens.

        Args:
            model: The MDLM model for inference
            tokenizer: BPE tokenizer with vocab attribute
            mask_token: The mask token id
            seq_len: Sequence length
            device: torch device
            char_width: Width per character in pixels
            cell_height: Height of each cell in pixels
            fps: Frames per second (default 10 fps = 0.1 sec per frame)
            target_width: Target window width (height auto-calculated)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.mask_token = mask_token
        self.seq_len = seq_len
        self.device = device
        self.char_width = char_width
        self.cell_height = cell_height
        self.fps = fps
        self.gap = 3  # Gap between cells

        # State
        self.history = None  # Will be populated after inference
        self.state = "idle"  # "idle", "ready", "playing"
        self.current_frame = 0
        self.playing = False
        self.prev_tokens = None

        # Layout
        self.margin = 30
        self.header_height = 80
        self.width = target_width

        # Calculate height based on average token width from vocabulary
        # Use average of all token lengths to estimate, with some padding for safety
        token_lengths = [len(tokenizer.vocab.get(i, '?')) for i in range(len(tokenizer.vocab))]
        avg_token_len = sum(token_lengths) / len(token_lengths) if token_lengths else 3
        avg_token_width = max(2, avg_token_len) * self.char_width + 12

        tokens_per_row = (self.width - 2 * self.margin) // (avg_token_width + self.gap)
        tokens_per_row = max(1, tokens_per_row)
        estimated_rows = (self.seq_len + tokens_per_row - 1) // tokens_per_row
        # Add 5% more rows as buffer for variable-width tokens
        estimated_rows = int(estimated_rows * 1.05)
        self.height = self.header_height + self.margin * 2 + estimated_rows * (cell_height + self.gap) + 60

        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("MDLM Denoising Visualization")

        # Fonts (scaled up)
        self.cell_font = pygame.font.SysFont('Monaco', 18)
        self.header_font = pygame.font.SysFont('Monaco', 24)
        self.title_font = pygame.font.SysFont('Monaco', 32, bold=True)
        self.button_font = pygame.font.SysFont('Monaco', 26, bold=True)

        self.clock = pygame.time.Clock()

        # Content area
        self.content_height = self.height - self.header_height - 2 * self.margin

        # Button positions (centered, larger)
        self.button_width = 240
        self.button_height = 60
        self.button_x = (self.width - self.button_width) // 2
        self.button_y = self.header_height + (self.content_height - self.button_height) // 2

        # "New Batch" button (shown when complete, larger)
        self.new_batch_btn_width = 180
        self.new_batch_btn_height = 48
        self.new_batch_btn_x = self.width - self.margin - self.new_batch_btn_width
        self.new_batch_btn_y = 16

    def is_button_hovered(self, mouse_pos):
        """Check if mouse is over the main Generate button."""
        mx, my = mouse_pos
        return (self.button_x <= mx <= self.button_x + self.button_width and
                self.button_y <= my <= self.button_y + self.button_height)

    def is_new_batch_hovered(self, mouse_pos):
        """Check if mouse is over the New Batch button."""
        mx, my = mouse_pos
        return (self.new_batch_btn_x <= mx <= self.new_batch_btn_x + self.new_batch_btn_width and
                self.new_batch_btn_y <= my <= self.new_batch_btn_y + self.new_batch_btn_height)

    def draw_button(self, x, y, w, h, text, hovered, font=None):
        """Draw a styled button with hover effect."""
        if font is None:
            font = self.button_font

        # Button shadow
        shadow_offset = 2
        pygame.draw.rect(self.screen, (10, 10, 15),
                        (x + shadow_offset, y + shadow_offset, w, h),
                        border_radius=10)

        # Button background
        btn_color = self.BUTTON_HOVER_COLOR if hovered else self.BUTTON_COLOR
        pygame.draw.rect(self.screen, btn_color, (x, y, w, h), border_radius=10)

        # Button border (subtle)
        border_color = (150, 200, 230) if hovered else (80, 95, 140)
        pygame.draw.rect(self.screen, border_color, (x, y, w, h), width=1, border_radius=10)

        # Button text
        text_color = self.BG_COLOR if hovered else self.BUTTON_TEXT_COLOR
        btn_text = font.render(text, True, text_color)
        btn_rect = btn_text.get_rect(center=(x + w // 2, y + h // 2))
        self.screen.blit(btn_text, btn_rect)

    def draw_idle_screen(self, mouse_pos):
        """Draw the idle screen with Generate button."""
        self.screen.fill(self.BG_COLOR)

        # Draw title
        title = self.title_font.render("MDLM Denoising", True, self.HEADER_COLOR)
        self.screen.blit(title, (self.margin, 15))

        # Draw Generate button (centered)
        hovered = self.is_button_hovered(mouse_pos)
        self.draw_button(self.button_x, self.button_y,
                        self.button_width, self.button_height,
                        "Generate", hovered)

        pygame.display.flip()

    def draw_loading_screen(self):
        """Draw loading screen during inference."""
        self.screen.fill(self.BG_COLOR)

        # Draw title
        title = self.title_font.render("MDLM Denoising", True, self.HEADER_COLOR)
        self.screen.blit(title, (self.margin, 10))

        # Draw loading text centered
        loading_text = self.title_font.render("Running inference...", True, self.HEADER_COLOR)
        loading_rect = loading_text.get_rect(center=(self.width // 2, self.height // 2))
        self.screen.blit(loading_text, loading_rect)

        pygame.display.flip()

    def run_inference(self):
        """Run inference and store history."""
        self.draw_loading_screen()
        pygame.event.pump()  # Keep window responsive

        self.history = sample_with_history(
            self.model,
            seq_len=self.seq_len,
            mask_token=self.mask_token,
            num_steps=100,
            temperature=1.0,
            device=self.device
        )

        self.state = "playing"
        self.current_frame = 0
        self.prev_tokens = None
        self.playing = True

    def get_token_str(self, token_id):
        """Get the raw token string."""
        if token_id == self.mask_token:
            return '<MASK>'
        return self.tokenizer.vocab.get(token_id, '?')

    def get_token_display(self, token_str):
        """Convert token string to displayable text."""
        if token_str == '<MASK>':
            return '?'
        # Character replacements for display
        replacements = {
            '\n': ' ',  # Newline as space
            '\t': ' ',  # Tab as space
            '"': '"',   # Smart quotes to regular
            '"': '"',
            ''': "'",
            ''': "'",
            '—': '-',   # Em-dash to hyphen
            '–': '-',   # En-dash to hyphen
            '…': '...',  # Ellipsis
        }
        display = ""
        for char in token_str:
            if char in replacements:
                display += replacements[char]
            elif ord(char) < 32:
                # Control characters - show as space
                display += ' '
            elif ord(char) > 126:
                # Extended ASCII/Unicode - try to show, fallback to ?
                display += char  # Let pygame try to render it
            else:
                display += char
        return display if display else '?'

    def is_space_token(self, token_id):
        """Check if a token is purely whitespace."""
        if token_id == self.mask_token:
            return False
        token_str = self.tokenizer.vocab.get(token_id, '')
        return token_str.strip() == '' and len(token_str) > 0

    def merge_tokens_into_words(self, tokens):
        """
        Merge adjacent tokens into word groups.
        Space tokens act as separators.
        Returns list of groups, where each group is a list of token indices.
        """
        groups = []
        current_group = []

        for i, token_id in enumerate(tokens):
            if token_id == self.mask_token:
                # Masks stay individual
                if current_group:
                    groups.append(current_group)
                    current_group = []
                groups.append([i])
            elif self.is_space_token(token_id):
                # Space token - end current group, add space as its own group
                if current_group:
                    groups.append(current_group)
                    current_group = []
                groups.append([i])
            else:
                # Regular token - check if it starts with space
                token_str = self.tokenizer.vocab.get(token_id, '')
                if token_str.startswith(' ') or token_str.startswith('\n'):
                    # Token starts with space - it's a new word
                    if current_group:
                        groups.append(current_group)
                    current_group = [i]
                else:
                    # Continue current word
                    current_group.append(i)

        if current_group:
            groups.append(current_group)

        return groups

    def get_group_display(self, tokens, group):
        """Get display string for a group of tokens."""
        display = ""
        for idx in group:
            token_str = self.get_token_str(tokens[idx])
            display += self.get_token_display(token_str)
        return display

    def get_group_width(self, display_str):
        """Calculate pixel width for a display string."""
        char_count = max(2, len(display_str))
        return char_count * self.char_width + 12

    def draw_mask_cell(self, x, y, width, height):
        """Draw a stylized mask cell with animated-looking pattern."""
        pygame.draw.rect(self.screen, self.MASK_BG_COLOR,
                        (x, y, width - 1, height - 1), border_radius=3)

        inner_margin = 2
        pygame.draw.rect(self.screen, (55, 45, 70),
                        (x + inner_margin, y + inner_margin,
                         width - 1 - 2*inner_margin, height - 1 - 2*inner_margin),
                        width=1, border_radius=2)

        char_surf = self.cell_font.render('?', True, self.MASK_COLOR)
        char_rect = char_surf.get_rect(center=(x + width//2, y + height//2))
        self.screen.blit(char_surf, char_rect)

    def compute_layout(self, tokens):
        """
        Compute layout with word-merged cells.
        Returns list of (group_indices, display_str, x, y, width, height, has_mask) tuples.
        """
        groups = self.merge_tokens_into_words(tokens)
        layout = []
        x = self.margin
        y = self.header_height + self.margin
        row_height = self.cell_height

        for group in groups:
            # Check if group contains any masks
            has_mask = any(tokens[idx] == self.mask_token for idx in group)

            if has_mask and len(group) == 1:
                # Single mask token
                display_str = '?'
                width = self.get_group_width(display_str)
            else:
                display_str = self.get_group_display(tokens, group)
                width = self.get_group_width(display_str)

            # Check if this is a space-only token - make it narrower
            is_space_only = len(group) == 1 and self.is_space_token(tokens[group[0]])
            if is_space_only:
                width = width // 2  # Half width for space tokens

            # Check if we need to wrap to next line
            if x + width > self.width - self.margin:
                x = self.margin
                y += row_height + self.gap

            layout.append((group, display_str, x, y, width, row_height, has_mask))
            x += width + self.gap

        return layout

    def draw_frame(self, frame_idx, mouse_pos=(0, 0)):
        """Draw a single frame with dynamic cell sizing."""
        self.screen.fill(self.BG_COLOR)

        tokens = self.history[frame_idx]
        progress = (len(self.history) - 1 - frame_idx) / max(1, len(self.history) - 1)
        is_complete = (frame_idx == len(self.history) - 1)

        # Draw title
        title = self.title_font.render("MDLM Denoising", True, self.HEADER_COLOR)
        self.screen.blit(title, (self.margin, 15))

        # Draw progress bar
        bar_x = self.margin
        bar_y = 58
        bar_width = 400
        bar_height = 10

        pygame.draw.rect(self.screen, self.PROGRESS_BAR_BG,
                        (bar_x, bar_y, bar_width, bar_height), border_radius=5)
        if progress > 0:
            pygame.draw.rect(self.screen, self.PROGRESS_BAR_FG,
                            (bar_x, bar_y, int(bar_width * progress), bar_height), border_radius=5)

        # Draw stats
        stats_y = 52
        step_text = f"Step {frame_idx}/{len(self.history)-1}"
        step_surf = self.header_font.render(step_text, True, (100, 105, 120))
        self.screen.blit(step_surf, (bar_x + bar_width + 20, stats_y))

        # Show "New Batch" button when complete
        if is_complete:
            hovered = self.is_new_batch_hovered(mouse_pos)
            self.draw_button(self.new_batch_btn_x, self.new_batch_btn_y,
                           self.new_batch_btn_width, self.new_batch_btn_height,
                           "New Batch", hovered, self.header_font)

        # Compute layout and draw tokens
        layout = self.compute_layout(tokens)

        for group, display_str, x, y, width, height, has_mask in layout:

            # Check if any token in this group was just unmasked
            was_just_unmasked = False
            if self.prev_tokens is not None:
                for idx in group:
                    if self.prev_tokens[idx] == self.mask_token and tokens[idx] != self.mask_token:
                        was_just_unmasked = True
                        break

            if has_mask and len(group) == 1:
                # Single mask token
                self.draw_mask_cell(x, y, width, height)
            else:
                # Normal cell (word group)
                if was_just_unmasked:
                    bg_color = self.NEWLY_UNMASKED_BG
                    text_color = self.NEWLY_UNMASKED_COLOR
                else:
                    bg_color = self.GRID_COLOR
                    text_color = self.TEXT_COLOR

                pygame.draw.rect(self.screen, bg_color,
                               (x, y, width - 1, height - 1),
                               border_radius=3)

                # Draw text
                text_surf = self.cell_font.render(display_str, True, text_color)
                text_rect = text_surf.get_rect(center=(x + width//2, y + height//2))
                self.screen.blit(text_surf, text_rect)

        pygame.display.flip()
        self.prev_tokens = tokens.copy()

    def run(self):
        """Main loop."""
        running = True
        frame_timer = 0
        frame_delay = 1000 // self.fps  # ms per frame

        while running:
            dt = self.clock.tick(60)  # 60 FPS for smooth UI
            mouse_pos = pygame.mouse.get_pos()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                        running = False
                    elif self.state == "playing" and self.history:
                        if event.key == pygame.K_SPACE:
                            self.playing = not self.playing
                        elif event.key == pygame.K_LEFT:
                            self.playing = False
                            self.current_frame = max(0, self.current_frame - 1)
                            self.prev_tokens = self.history[max(0, self.current_frame - 1)] if self.current_frame > 0 else None
                        elif event.key == pygame.K_RIGHT:
                            self.playing = False
                            self.current_frame = min(len(self.history) - 1, self.current_frame + 1)
                        elif event.key == pygame.K_r:
                            self.current_frame = 0
                            self.prev_tokens = None
                            self.playing = True
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        if self.state == "idle":
                            if self.is_button_hovered(mouse_pos):
                                self.run_inference()
                        elif self.state == "playing" and self.history:
                            # Check if "New Batch" button clicked (only visible when complete)
                            is_complete = (self.current_frame == len(self.history) - 1)
                            if is_complete and self.is_new_batch_hovered(mouse_pos):
                                self.run_inference()

            # Draw based on state
            if self.state == "idle":
                self.draw_idle_screen(mouse_pos)
            elif self.state == "playing" and self.history:
                # Auto-advance if playing
                if self.playing:
                    frame_timer += dt
                    if frame_timer >= frame_delay:
                        frame_timer = 0
                        if self.current_frame < len(self.history) - 1:
                            self.current_frame += 1
                        else:
                            self.playing = False  # Stop at end

                self.draw_frame(self.current_frame, mouse_pos)

        pygame.quit()


# ============================================================================
# Main
# ============================================================================

def main():
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    print(f"Using device: {device}")

    # Load BPE tokenizer
    TOKENIZER_PATH = './tokenizer_bpe.json'
    tokenizer = SimpleBPE().load(TOKENIZER_PATH)

    final_vocab_size = len(tokenizer.vocab)
    MASK_TOKEN = tokenizer.special_tokens.get('<MASK>')
    if MASK_TOKEN is None:
        raise ValueError("Tokenizer missing <MASK> token")

    print(f'Mask token id is {MASK_TOKEN}')
    print(f'Final vocab size is {final_vocab_size}')

    # Hyperparameters (must match training - from sampling.ipynb)
    block_size = 512
    n_embd = 512
    n_head = 4
    n_blocks = 6

    # Load model
    CKPT_PATH = './ckpt/latest.pt'

    model = MDLM(
        vocab_size=final_vocab_size,
        n_embd=n_embd,
        n_head=n_head,
        n_block=n_blocks,
        block_size=block_size,
    ).to(device)

    if os.path.exists(CKPT_PATH):
        print(f"Loading model weights from {CKPT_PATH}...")
        checkpoint = torch.load(CKPT_PATH, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded successfully (from Epoch {checkpoint['epoch']+1})")
    else:
        print(f"Warning: No checkpoint found at {CKPT_PATH}. Using random initialization.")

    seq_len = 512

    print("\nStarting visualization...")
    print("Click 'Generate' button to run inference")
    print("Controls: SPACE=Play/Pause, LEFT/RIGHT=Step, R=Restart, Q=Quit")

    # Run visualization
    visualizer = MDLMVisualizer(
        model=model,
        tokenizer=tokenizer,
        mask_token=MASK_TOKEN,
        seq_len=seq_len,
        device=device,
        char_width=14,
        cell_height=32,
        fps=10,
        target_width=1400
    )
    visualizer.run()


if __name__ == "__main__":
    main()
