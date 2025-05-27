import os
import re
import glob
from torch.utils.tensorboard import SummaryWriter

TB_ROOT = "logs/tensorboard"
_CACHED_RUN_DIR = None

def _next_index() -> int:
    """
    Restituisce N tale che runN è l’indice successivo libero.
    Cerca tutte le cartelle run* e prende il massimo +1.
    """
    runs = [re.search(r"run(\d+)", p) for p in glob.glob(f"{TB_ROOT}/run*")]
    nums = [int(m.group(1)) for m in runs if m]
    return max(nums, default=0) + 1

def get_tb_writer() -> SummaryWriter:
    """
    Restituisce un SummaryWriter che punta a logs/tensorboard/runX[_label].
    Alla prima chiamata chiede via input() il label della run,
    e memorizza internamente la cartella scelta. Le chiamate successive
    riutilizzano la stessa cartella senza ripromptare.
    """
    global _CACHED_RUN_DIR
    os.makedirs(TB_ROOT, exist_ok=True)

    if _CACHED_RUN_DIR is None:
        last_idx = _next_index() - 1  # 0 se non esiste alcuna run
        default = last_idx or 1
        label = input(f"Label nuova run?  [invio = continua run{default}] > ").strip()

        if label:
            # nuova run: usa next index e aggiunge l’etichetta
            run_dir = f"{TB_ROOT}/run{_next_index()}_{label.replace(' ', '_')}"
        else:
            # continua la run esistente (o crea run1)
            run_dir = f"{TB_ROOT}/run{default}"

        print("TensorBoard dir:", run_dir)
        _CACHED_RUN_DIR = run_dir

    return SummaryWriter(log_dir=_CACHED_RUN_DIR)
