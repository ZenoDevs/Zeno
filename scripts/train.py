from src.utils.tb import get_tb_writer        # <— nuovo import
from src.utils.logger import get_logger
from src.env.gridworld import GridWorld
from src.agents.zeno   import ZenoAgent
from src.rl.trainer    import ReinforceTrainer

if __name__ == "__main__":
    log     = get_logger()
    tbw     = get_tb_writer()                 # ottieni il writer con logica runN

    env   = GridWorld()
    # qui specifichiamo state_dim=4
    zeno  = ZenoAgent(state_dim=4, lr=1e-2)
    loop  = ReinforceTrainer(env, zeno,
                             episodes = 2000,
                             max_steps= 50,    # più rapido
                             gamma    = 0.99)

    loop.run()