from src.utils.tb import get_tb_writer        # <— nuovo import
from src.utils.logger import get_logger
from src.env.gridworld import GridWorld
from src.agents.zeno   import ZenoAgent
from src.rl.trainer    import ReinforceTrainer

if __name__ == "__main__":
    log     = get_logger()
    tbw     = get_tb_writer()                 # ottieni il writer con logica runN

    env   = GridWorld()
    agent = ZenoAgent()

    trainer = ReinforceTrainer(env, agent,
                               episodes  = 1000,
                               gamma     = 0.99,
                               max_steps = 100,
                               tb_writer = tbw)   # passa il writer

    trainer.run()
