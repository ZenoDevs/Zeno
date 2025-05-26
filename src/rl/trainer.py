# src/rl/trainer.py  (versione Fixata)
import torch, itertools
from torch.utils.tensorboard import SummaryWriter
from ..utils.logger import get_logger
log = get_logger(__name__)

class ReinforceTrainer:
    def __init__(self, env, agent,
                 episodes  = 1000,
                 gamma     = 0.99,
                 max_steps = 100,
                 tb_writer = None,
                 tb_flush  = 50):
        self.env, self.agent = env, agent
        self.episodes, self.gamma = episodes, gamma
        self.max_steps   = max_steps
        self.tb_flush    = tb_flush
        self.tb = tb_writer or SummaryWriter("logs/tensorboard/run1")


    # --------------------------------------------------
    def run(self):
        for ep in range(1, self.episodes + 1):
            state = self.env.reset()
            logbuf, rewbuf, steps, done = [], [], 0, False

            while not done and steps < self.max_steps:
                action, logp = self.agent.act(state)
                state, r, done = self.env.step(action)

                logbuf.append(logp)
                rewbuf.append(r)
                steps += 1

            # penalità se non ha ottenuto l'acqua entro il limite
            if not done:
                rewbuf[-1] -= 0.1

            # ritorni scontati
            G = 0; returns = []
            for r in reversed(rewbuf):
                G = r + self.gamma * G
                returns.insert(0, G)
            returns = torch.tensor(returns)

            loss = -(torch.stack(logbuf) * returns).sum()
            self.agent.opt.zero_grad(); loss.backward(); self.agent.opt.step()

            ep_reward = sum(rewbuf)
            self.tb.add_scalar("reward", ep_reward, ep)
            self.tb.add_scalar("steps",  steps,      ep)
            if ep % self.tb_flush == 0:
                self.tb.flush()          # <-- forza scrittura

            log.info("Ep %04d | reward %.2f | steps %d", ep, ep_reward, steps)

        self.tb.flush(); self.tb.close()
        log.info("Training finished.")
