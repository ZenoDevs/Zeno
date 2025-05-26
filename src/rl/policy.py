# src/rl/policy.py
import torch
import torch.nn as nn

class PolicyNet(nn.Module):
    """
    Embed token-id + stato (x,y,thirst) e produce logits sui token.
    - state_dim  = 3  (x, y, thirst_flag)
    - vocab_size = len(vocab)
    """
    def __init__(self, state_dim: int, vocab_size: int,
                 emb_dim: int = 16, hidden: int = 32):
        super().__init__()
        self.state_fc = nn.Linear(state_dim, hidden)
        self.embed    = nn.Embedding(vocab_size, emb_dim)  # (non serve ora, ma utile dopo)
        self.head     = nn.Linear(hidden, vocab_size)

    def forward(self, state):
        """
        state: tensor (batch, 3)
        ritorna: logits (batch, vocab_size)
        """
        x = torch.tanh(self.state_fc(state))
        return self.head(x)
