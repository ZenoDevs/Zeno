import numpy as np
from ..utils.logger import get_logger
log = get_logger(__name__)

class GridWorld:
    """
    Griglia 3×3. Celle (0,0)…(2,2).
    - Zeno parte a (0,0).
    - Madre fissa a (2,2) – non si muove.
    Stato = (pos_x, pos_y, sete_bool)
    """
    SIZE = 3

    def __init__(self):
        self.reset()

    def reset(self):
        self.pos = np.array([0, 0])   # angolo in alto a sinistra
        self.thirst = True
        self.terminated = False
        log.info("Env reset: Zeno @%s  sete=%s", self.pos, self.thirst)
        return self._state()

    def step(self, action: str):
        """
        action:
          'speak:<token>'            -> token emesso
        Ritorna (state, reward, terminated)
        """
        reward = 0.0

        if action.startswith("speak:"):
            token = action.split(":", 1)[1]
            from ..agents.mother import mother_reply
            given, reward = mother_reply(token, self.thirst)
            if given:
                self.thirst = False
                self.terminated = True
        else:
            reward = -0.1

        return self._state(), reward, self.terminated

    # ---------- helpers ----------
    def _state(self):
        # vettore NumPy per la NN (x, y, thirsty_flag)
        return np.array([*self.pos, int(self.thirst)], dtype=np.float32)

   
