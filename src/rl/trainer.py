# src/rl/trainer.py
import torch
from torch.utils.tensorboard import SummaryWriter
from ..utils.logger import get_logger
from ..utils.tb     import get_tb_writer

log = get_logger(__name__)

class ReinforceTrainer:
    def __init__(self, env, agent,
                 episodes  : int   = 1000,
                 gamma     : float = 0.99,
                 max_steps : int   = 100,
                 tb_flush  : int   = 50):
        """
        env       : ambiente GridWorld
        agent     : ZenoAgent o simile
        episodes  : numero di episodi di training
        gamma     : discount factor
        max_steps : passo-max per episodio
        tb_flush  : ogni quanti episodi fare flush()
        """

        self.env       = env
        self.agent     = agent
        self.episodes  = episodes
        self.gamma     = gamma
        self.max_steps = max_steps
        self.tb_flush  = tb_flush

        # *** qui auto–creo il writer ***
        # stamperà a schermo "Label nuova run? [invio = continua runN] >"
        self.tb = get_tb_writer()

    def run(self):
        for ep in range(1, self.episodes + 1):
            state = self.env.reset()
            logbuf, rewbuf, steps, done = [], [], 0, False

            while not done and steps < self.max_steps:
                action, logp   = self.agent.act(state)
                state, r, done = self.env.step(action)

                logbuf.append(logp)
                rewbuf.append(r)
                steps += 1

            if not done:
                rewbuf[-1] -= 0.1

            # calcolo ritorni
            G, returns = 0, []
            for r in reversed(rewbuf):
                G = r + self.gamma * G
                returns.insert(0, G)
            returns = torch.tensor(returns, dtype=torch.float32)

            # backprop
            loss = -(torch.stack(logbuf) * returns).sum()
            self.agent.opt.zero_grad()
            loss.backward()
            self.agent.opt.step()

            ep_reward = sum(rewbuf)
            self.tb.add_scalar("reward", ep_reward, ep)
            self.tb.add_scalar("steps",  steps,      ep)
            if ep % self.tb_flush == 0:
                self.tb.flush()

            log.info("Ep %04d | reward %.2f | steps %d", ep, ep_reward, steps)

        self.tb.flush()
        self.tb.close()
        log.info("Training finished.")
