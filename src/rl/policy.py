# src/rl/policy.py
import torch
import torch.nn as nn

class PolicyNet(nn.Module):
    """
    Trasforma lo stato (batch×state_dim) in logits per ciascuno dei vocab_size token.

    Parameters
    ----------
    vocab_size : int
        Numero di parole nel vocabolario.
    state_dim : int
        Dimensione del vettore di stato (qui = 2: thirst_flag, hunger_flag).
    emb_dim : int, default=16
        Dimensione dell’Embedding (futura estensione).
    hidden : int, default=32
        Dimensione dello strato nascosto.
    """
    def __init__(self, vocab_size: int, state_dim: int,
                 emb_dim: int = 16, hidden: int = 32):
        super().__init__()
        self.state_fc = nn.Linear(state_dim, hidden)
        self.embed    = nn.Embedding(vocab_size, emb_dim)
        self.head     = nn.Linear(hidden, vocab_size)

    def forward(self, state):
        x = torch.tanh(self.state_fc(state))
        return self.head(x)
