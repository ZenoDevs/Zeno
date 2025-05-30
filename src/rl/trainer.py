# src/rl/trainer.py
import os
import re
import glob
import torch
import statistics
from torch.utils.tensorboard import SummaryWriter
from ..utils.logger import get_logger
from ..utils.tb import get_tb_writer

log = get_logger(__name__)

class ReinforceTrainer:
    def __init__(self,
                 env,
                 agent,
                 episodes   = 1000,
                 gamma      = 0.99,
                 max_steps  = 100,
                 tb_flush   = 50,
                 ckpt_dir   = "checkpoints"):
        self.env        = env
        self.agent      = agent
        self.episodes   = episodes
        self.gamma      = gamma
        self.max_steps  = max_steps
        self.tb_flush   = tb_flush

        # TensorBoard writer
        self.tb = get_tb_writer()

        # checkpoint directory
        self.ckpt_dir = ckpt_dir
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # metriche
        self.running_baseline = 0.0
        self._succ_buffer     = []
        self._steps_success   = []

        # se esiste un checkpoint, caricalo
        latest = self._find_latest_ckpt()
        if latest is not None:
            self._load_checkpoint(latest)
            log.info("⟳ Riprendo da checkpoint ep=%d", latest)
            self.start_ep = latest
        else:
            self.start_ep = 0

    # ---------------- utils per checkpoint ------------------

    def _find_latest_ckpt(self):
        """Cerca tutti i file ckpt_ep{N}.pth e restituisce il N massimo."""
        pattern = os.path.join(self.ckpt_dir, "ckpt_ep*.pth")
        files = glob.glob(pattern)
        epochs = []
        for f in files:
            name = os.path.basename(f)
            m = re.match(r"ckpt_ep(\d+)\.pth$", name)
            if m:
                epochs.append(int(m.group(1)))
        return max(epochs) if epochs else None

    def _load_checkpoint(self, ep):
        """Carica pesi modello e ottimizzatore dal checkpoint ep."""
        model_path = os.path.join(self.ckpt_dir, f"ckpt_ep{ep}.pth")
        opt_path   = os.path.join(self.ckpt_dir, f"opt_ep{ep}.pth")
        log.info("⟳ Carico modello da %s", model_path)
        model_sd = torch.load(model_path, map_location="cpu")
        self.agent.net.load_state_dict(model_sd)
        log.info("⟳ Carico ottimizzatore da %s", opt_path)
        opt_sd = torch.load(opt_path, map_location="cpu")
        self.agent.opt.load_state_dict(opt_sd)

    def _save_checkpoint(self, ep):
        """Salva modello e ottimizzatore sotto ckpt_ep{ep}.pth / opt_ep{ep}.pth."""
        model_path = os.path.join(self.ckpt_dir, f"ckpt_ep{ep}.pth")
        opt_path   = os.path.join(self.ckpt_dir, f"opt_ep{ep}.pth")
        torch.save(self.agent.net.state_dict(), model_path)
        torch.save(self.agent.opt.state_dict(),   opt_path)
        log.info("✔ Checkpoint salvato ep=%d", ep)

    # --------------------- training loop --------------------

    def run(self):
        # riparti dall'epoca successiva all'ultimo checkpoint
        for ep in range(self.start_ep + 1, self.episodes + 1):
            state = self.env.reset()
            logbuf, rewbuf, steps, done = [], [], 0, False

            # rollout
            while not done and steps < self.max_steps:
                action, logp   = self.agent.act(state)
                state, r, done = self.env.step(action)
                logbuf.append(logp)
                rewbuf.append(r)
                steps += 1

            # penalità se non terminato
            if not done:
                rewbuf[-1] -= 0.1

            # calcolo ritorni scontati
            G, returns = 0.0, []
            for r in reversed(rewbuf):
                G = r + self.gamma * G
                returns.insert(0, G)
            returns = torch.tensor(returns)

            # baseline e advantage
            total_reward = returns.sum().item()
            alpha = 0.1
            self.running_baseline = (1 - alpha) * self.running_baseline + alpha * total_reward
            advantages = returns - self.running_baseline

            # entropy bonus
            logps   = torch.stack(logbuf)
            entropy = - (logps * logps.exp()).sum() / len(logps)
            beta    = 1e-3

            # loss e update
            loss = -(logps * advantages).sum() - beta * entropy
            self.agent.opt.zero_grad()
            loss.backward()
            self.agent.opt.step()

            # metriche di successo
            is_success = float(total_reward > 0)
            self._succ_buffer.append(is_success)
            if is_success:
                self._steps_success.append(steps)

            # logging su TensorBoard
            self.tb.add_scalar("reward",      total_reward, ep)
            self.tb.add_scalar("steps",       steps,        ep)
            self.tb.add_scalar("entropy",     entropy,      ep)

            # flush / checkpoint ogni tb_flush epoche
            if ep % self.tb_flush == 0:
                sr = statistics.mean(self._succ_buffer)
                self.tb.add_scalar("succ_rate",     sr, ep)
                if self._steps_success:
                    self.tb.add_scalar("steps_success",
                                      statistics.mean(self._steps_success),
                                      ep)
                self._save_checkpoint(ep)
                self._succ_buffer.clear()
                self._steps_success.clear()
                self.tb.flush()

            log.info("Ep %04d | R=%.2f | S=%d | Baseline=%.2f",
                     ep, total_reward, steps, self.running_baseline)

        # ultimo checkpoint a fine training
        self._save_checkpoint(self.episodes)
        self.tb.close()
        log.info("🏁 Training finished.")
