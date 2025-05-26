import os, re, glob
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

TB_ROOT = "logs/tensorboard"

def _next_index() -> int:
    """Restituisce N tale che runN esiste per ultimo."""
    runs = [re.search(r"run(\d+)", p) for p in glob.glob(f"{TB_ROOT}/run*")]
    nums = [int(m.group(1)) for m in runs if m]
    return max(nums, default=0) + 1

def get_tb_writer() -> SummaryWriter:
    os.makedirs(TB_ROOT, exist_ok=True)
    last_idx = _next_index() - 1           # 0 se non c'è nulla

    label = input(f"Label nuova run?  [invio = continua run{last_idx or 1}] > ").strip()

    if label:                        # --- nuovo run ---
        run_dir = f"{TB_ROOT}/run{_next_index()}_{label.replace(' ', '_')}"
    else:                            # --- continua run esistente (o crea run1) ---
        run_dir = f"{TB_ROOT}/run{last_idx or 1}"

    print("TensorBoard dir:", run_dir)
    return SummaryWriter(log_dir=run_dir)
