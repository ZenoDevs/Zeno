"""
Madre: logica di ricompensa hard-coded.
Se Zeno ha sete e pronuncia token 'acqua' => consegna acqua e +1 reward.
Altrimenti 0.
"""

from ..utils.logger import get_logger
log = get_logger(__name__)

TARGET_TOKEN = "acqua"

def mother_reply(token: str, still_thirsty: bool):
    if still_thirsty and token == TARGET_TOKEN:
        log.info("Madre: ho sentito '%s' → do acqua (+1)", token)
        return True, 1.0
    # nessun reward
    return False, 0.0

