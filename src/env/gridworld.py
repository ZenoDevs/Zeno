# src/env/gridworld.py
import numpy as np, random
from ..utils.logger import get_logger
log = get_logger(__name__)

class GridWorld:
    """
    Ambiente “parlante” senza movimento: ogni episodio
    l’agente ha o sete o fame e deve parlare il token giusto.
    Stato = (thirst_flag, hunger_flag)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        # scegli casualmente un bisogno: "thirst" o "hunger"
        self.need = random.choice(["thirst", "hunger"])
        self.thirst = (self.need == "thirst")
        self.hunger = (self.need == "hunger")
        self.terminated = False

        log.info("Zeno reset: need=%s", self.need)
        return self._state()

    def step(self, action: str):
        """
        action: 'speak:<token>'
        Ritorna (state, reward, done)
        """
        reward = 0.0

        if action.startswith("speak:"):
            token = action.split(":", 1)[1]
            from ..agents.mother import mother_reply
            given, reward = mother_reply(token, self.need)
            if given:
                self.terminated = True
        else:
            # qualsiasi altra “azione” è inutile
            reward = -0.1

        return self._state(), reward, self.terminated

    def _state(self):
        # restituisci solo i due flag di bisogno
        return np.array([int(self.thirst), int(self.hunger)],
                        dtype=np.float32)
