"""
Madre: logica di ricompensa hard-coded.
Se Zeno ha sete e pronuncia token 'acqua' => consegna acqua e +1 reward.
Altrimenti 0.
"""

from ..utils.logger import get_logger
log = get_logger(__name__)

TARGET_TOKEN = "acqua"

def mother_reply(token: str, is_thirsty: bool, is_hungry: bool):
    """
    Reward = 1 se:
      - is_thirsty and token=='acqua'
      - is_hungry  and token=='cibo'
    """
    if is_thirsty and token == "acqua":
        log.info("Madre: ho sentito 'acqua' → do acqua (+1)")
        return True, 1.0
    if is_hungry and token == "cibo":
        log.info("Madre: ho sentito 'cibo' → do cibo (+1)")
        return True, 1.0
    return False, 0.0