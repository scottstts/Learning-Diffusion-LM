"""
Configuration for MDLM Sampling Visualization
"""

# ============================================================================
# Paths
# ============================================================================
TOKENIZER_PATH = '.data/tokenizer_bpe.json'
CKPT_PATH = './ckpt/latest.pt'

# ============================================================================
# Model Hyperparameters
# ============================================================================
BLOCK_SIZE = 1024
N_EMBD = 768
N_HEAD = 12
N_BLOCKS = 12

# ============================================================================
# Sampling Parameters
# ============================================================================
SEQ_LEN = BLOCK_SIZE
NUM_STEPS = 100
TOP_P = 0.9
DEFAULT_TEMPERATURE = 1.0
TEMP_MIN = 0.0
TEMP_MAX = 1.0
TEMP_STEP = 0.1

# ============================================================================
# Visualization Layout
# ============================================================================
TARGET_WIDTH = 1400
CHAR_WIDTH = 10
CELL_HEIGHT = 22
FPS = 10
GAP = 2
MARGIN = 30
HEADER_HEIGHT = 80
SLIDER_HEIGHT = 50
AVG_CELL_WIDTH = 50

# Button dimensions
BUTTON_WIDTH = 240
BUTTON_HEIGHT = 60
NEW_BATCH_BTN_WIDTH = 180
NEW_BATCH_BTN_HEIGHT = 48

# Font sizes
CELL_FONT_SIZE = 13
HEADER_FONT_SIZE = 20
TITLE_FONT_SIZE = 28
BUTTON_FONT_SIZE = 22

# ============================================================================
# Colors (RGB tuples)
# ============================================================================
BG_COLOR = (15, 15, 20)
GRID_COLOR = (35, 38, 48)
MASK_BG_COLOR = (45, 35, 55)
MASK_COLOR = (180, 130, 200)
TEXT_COLOR = (220, 220, 230)
NEWLY_UNMASKED_COLOR = (100, 220, 150)
NEWLY_UNMASKED_BG = (30, 60, 45)
HEADER_COLOR = (130, 200, 235)
BUTTON_COLOR = (70, 85, 130)
BUTTON_HOVER_COLOR = (100, 180, 220)
BUTTON_TEXT_COLOR = (240, 240, 245)
PROGRESS_BAR_BG = (30, 32, 40)
PROGRESS_BAR_FG = (100, 180, 220)
SLIDER_BG = (40, 45, 55)
SLIDER_FG = (100, 180, 220)
SLIDER_KNOB = (150, 200, 240)
