# Zeno – Need-Driven Language Emergence

Zeno è un sandbox di ricerca in cui un agente virtuale impara a usare il linguaggio **per soddisfare bisogni primari** (sete, fame, ecc.).  
Il linguaggio non viene fornito “già saputo”: emerge gradualmente perché l’agente riceve ricompense solo quando comunica in modo utile.

---

## Scope

| Stadio | Obiettivo principale           | Esempio di frase target           |
| ------ | ------------------------------ | --------------------------------- |
| 1      | Bisogno singolo, token singolo | `acqua`                           |
| 2      | Disambiguazione                | `acqua blu`                       |
| 3      | Bisogni multipli               | `voglio acqua blu e cibo`         |
| 4      | Frasi articolate               | `voglio acqua blu perché ho sete` |
| 5      | Dialogo a turni / cooperazione | `vai a destra, lì c'è il cibo`    |

---

## Project structure

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
|   |   |── tb.py
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
```

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/train.py --config configs/base.yaml
tensorboard --logdir logs/tensorboard

---

## 2. Guida rapida “Folder & File” (cosa serve a cosa)

| Percorso                              | A cosa serve – **parti da qui se…**                                                                        |
| ------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| **configs/**                          | YAML con hyper-parametri. Crea file diversi per esperimenti.                                               |
| **scripts/train.py**                  | Entry-point CLI: legge un config, avvia trainer. PUNTO DI PARTENZA per far girare il prototipo.            |
| **scripts/visualize.py**              | Replay grafico/ascii di un episodio già salvato. Utile per debugging.                                      |
| **src/env/gridworld.py**              | Logica dell’ambiente: griglia, stato sete/fame, `step()` e `reset()`. È il **primo file da implementare**. |
| **src/agents/zeno.py**                | Policy dell’agente che impara. Inizia con scelte random → poi rete PyTorch.                                |
| **src/agents/mother.py**              | “Madre” che valuta frasi e restituisce reward. All’inizio è hard-coded.                                    |
| **src/rl/policy.py** / **trainer.py** | Algoritmo RL generico (REINFORCE o PPO). Puoi usare un’implementazione minimale qui.                       |
| **src/utils/logger.py**               | Wrapper su `logging` + colori; centralizza tutti i log.                                                    |
| **src/utils/tokenization.py**         | Mapping parola ↔ ID; in futuro BPE/SentencePiece.                                                          |
| **src/utils/visualization.py**        | Funzioni per disegnare la griglia con Rich o PyGame.                                                       |
| **notebooks/**                        | Playground esplorativo: prova reward, grafici veloci.                                                      |
| **logs/**                             | Output runtime (stdout, metriche CSV, TensorBoard).                                                        |
| **checkpoints/**                      | Modelli `.pt` salvati dal trainer.                                                                         |
| **data/vocab.json**                   | Lista token editabile a mano; punto centrale per aggiungere vocaboli.                                      |

---

### **Da dove partire concretamente**

1. **Implementa `src/env/gridworld.py`** con un ambiente 3×3, stato `sete`, reward su token `"acqua"`.
2. **Scrivi `src/agents/mother.py`** hard-coded: se token == `"acqua"` → reward; altrimenti 0.
3. **Implementa un trainer minimale in `src/rl/trainer.py`** che usa REINFORCE.
4. **Lancia `scripts/train.py`** e guarda i log / TensorBoard.
5. Quando vedi il reward salire → passa allo Step 2 della roadmap (disambiguation).

---

Con queste note hai README pronto e una spiegazione chiara di ogni pezzo.  
Se vuoi raffinare qualcosa o hai domande su un file specifico, dimmi pure!
