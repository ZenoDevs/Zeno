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
        self.pos = np.array([0, 0])
        # scegli a caso un bisogno: 0 = sete, 1 = fame
        if np.random.rand() < 0.5:
            self.need = "thirst"
        else:
            self.need = "hunger"
        # due flag booleane 
        self.thirst = (self.need == "thirst")
        self.hunger = (self.need == "hunger")
        self.terminated = False
        log.info("Zeno reset: pos=%s, need=%s", self.pos, self.need)
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
            given, reward = mother_reply(token, self.thirst, self.hunger)
            if given:
                self.thirst = False
                self.hunger = False
                self.terminated = True
        else:
            reward = -0.1

        return self._state(), reward, self.terminated

    # ---------- helpers ----------
    def _state(self):
        # [x, y, thirst_flag, hunger_flag]
        return np.array([*self.pos,
                         int(self.thirst),
                         int(self.hunger)],
                        dtype=np.float32)
   
