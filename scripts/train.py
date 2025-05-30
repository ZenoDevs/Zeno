# scripts/train.py

from src.utils.logger import get_logger
from src.utils.tb import get_tb_writer
from src.env.gridworld import GridWorld
from src.agents.zeno import ZenoAgent
from src.rl.trainer import ReinforceTrainer

if __name__ == "__main__":
    log = get_logger()

    # crea env + agente
    env = GridWorld()
    zeno = ZenoAgent(state_dim=2, lr=1e-2)

    # interazione con utente
    try:
        episodes_input = input("Quante epoche totali vuoi raggiungere? (default: 2000) > ").strip()
        episodes = int(episodes_input) if episodes_input else 2000
    except ValueError:
        print("Input non valido. Uso default: 2000 epoche.")
        episodes = 2000

    # avvia il trainer
    loop = ReinforceTrainer(
        env, zeno,
        episodes=episodes,
        max_steps=50,
        gamma=0.99
    )
    loop.run()
