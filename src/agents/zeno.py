# src/agents/zeno.py
import torch, random
from ..rl.policy import PolicyNet
from ..utils.tokenization import TOKEN2ID, ID2TOKEN
from ..utils.logger import get_logger
log = get_logger(__name__)

DEVICE = "cpu"  # o "cuda" se disponibile

class ZenoAgent:
    def __init__(self, state_dim: int = 2, lr: float = 1e-2):
        self.vocab_size = len(TOKEN2ID)
        # PRIMO argomento = vocab_size, SECONDO = state_dim
        self.net = PolicyNet(self.vocab_size, state_dim).to(DEVICE)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

    def act(self, state):
        """
        state: np.array([thirst_flag, hunger_flag])
        Ritorna (action_str, log_prob)
        """
        state_t = torch.tensor(state, dtype=torch.float32,
                               device=DEVICE).unsqueeze(0)
        logits = self.net(state_t)
        distr  = torch.distributions.Categorical(logits=logits)
        token_id = distr.sample()
        logprob  = distr.log_prob(token_id)

        token = ID2TOKEN[token_id.item()]
        return f"speak:{token}", logprob
