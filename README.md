# Zeno – Need-Driven Language Emergence (Training on Italian)

Zeno is a research sandbox where a virtual agent learns to use language **to satisfy primary needs** (thirst, hunger, etc.).  
The language is **not** provided “pre-wired”: it emerges gradually because the agent only receives rewards when it communicates usefully—in **Italian**.

---

## Scope

| Stage | Main Goal                         | Example Target Phrase             |
| ----- | --------------------------------- | --------------------------------- |
| 1     | Single need, single token         | `acqua`                           |
| 2     | Disambiguation                    | `acqua blu`                       |
| 3     | Multiple needs                    | `voglio acqua blu e cibo`         |
| 4     | Articulated sentences             | `voglio acqua blu perché ho sete` |
| 5     | Turn-based dialogue / cooperation | `vai a destra, lì c’è il cibo`    |

---

## Project Structure

```text
zeno/
├── README.md
├── requirements.txt
├── pyproject.toml
│
├── configs/
│   ├── base.yaml
│   └── grid_3x3.yaml
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── visualize.py
│
├── src/
│   ├── env/
│   │   └── gridworld.py
│   ├── agents/
│   │   ├── zeno.py
│   │   └── mother.py
│   ├── rl/
│   │   ├── policy.py
│   │   └── trainer.py
│   ├── utils/
│   │   ├── logger.py
│   │   ├── tb.py
│   │   ├── tokenization.py
│   │   └── visualization.py
│   └── main.py
│
├── notebooks/
│   └── playground.ipynb
├── logs/
├── checkpoints/
└── data/
    └── vocab.json

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m scripts.train --config configs/base.yaml
tensorboard --logdir logs/tensorboard


| Path                                  | Purpose                                                                                                               |
| ------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| **configs/**                          | YAML files for hyper-parameters. Create a new config for each experiment.                                             |
| **scripts/train.py**                  | CLI entry-point: loads a config, instantiates the trainer & agent. **Your starting point for running the prototype.** |
| **scripts/visualize.py**              | Graphical/ASCII replay of a saved episode—useful for debugging.                                                       |
| **src/env/gridworld.py**              | Environment logic: 3×3 grid, thirst/hunger flags, `step()` & `reset()`. **First file to implement.**                  |
| **src/agents/zeno.py**                | Agent’s policy: initially random, then a PyTorch network.                                                             |
| **src/agents/mother.py**              | “Mother” oracle: hard-coded reward logic.                                                                             |
| **src/rl/policy.py** & **trainer.py** | Core RL algorithm (REINFORCE/PPO). A minimal implementation here.                                                     |
| **src/utils/logger.py**               | Colored‐output `logging` wrapper; centralizes all logs.                                                               |
| **src/utils/tb.py**                   | Factory for TensorBoard `SummaryWriter` with automatic run-naming.                                                    |
| **src/utils/tokenization.py**         | Word ↔ ID mapping; replaceable later with BPE/SentencePiece.                                                          |
| **src/utils/visualization.py**        | Grid‐drawing utilities (Rich, PyGame).                                                                                |
| **notebooks/**                        | Exploratory notebook for quick reward tests and plots.                                                                |
| **logs/**                             | Runtime outputs (stdout, CSV metrics, TensorBoard events).                                                            |
| **checkpoints/**                      | Saved model `.pt` checkpoints.                                                                                        |
| **data/vocab.json**                   | Editable vocabulary file (list of tokens).                                                                            |
```
