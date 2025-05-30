from ..utils.logger import get_logger
log = get_logger(__name__)

# mappa bisogno → token giusto
TARGET_MAP = {
    "thirst": "acqua",
    "hunger": "cibo",
}

def mother_reply(token: str, need: str) -> tuple[bool, float]:
    """
    Restituisce (True, 1.0) se token == TARGET_MAP[need],
    cioè:
      - need == "thirst" e token == "acqua"
      - need == "hunger" e token == "cibo"
    Altrimenti (False, 0.0).
    """
    target = TARGET_MAP.get(need)
    if token == target:
        log.info("Madre (need=%s): ho sentito '%s' → +1", need, token)
        return True, 1.0
    return False, 0.0