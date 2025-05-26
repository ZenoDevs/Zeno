# src/agents/zeno.py
import torch, random
from ..rl.policy import PolicyNet
from ..utils.tokenization import TOKEN2ID, ID2TOKEN
from ..utils.logger import get_logger
log = get_logger(__name__)

DEVICE = "cpu"  # metti "cuda" se hai GPU

class ZenoAgent:
    def __init__(self, state_dim=3, lr=1e-2):
        self.vocab_size = len(TOKEN2ID)
        self.net = PolicyNet(state_dim, self.vocab_size).to(DEVICE)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

    def act(self, state):
        """
        - prende lo stato (np.array) dal GridWorld
        - produce token_id ~ π(.|state)
        - ritorna  action_str  per l'env e  log_prob  per REINFORCE
        """
        state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        logits  = self.net(state_t)
        distr   = torch.distributions.Categorical(logits=logits)
        token_id = distr.sample()
        logprob  = distr.log_prob(token_id)

        token = ID2TOKEN[token_id.item()]
        action = f"speak:{token}" 
        return action, logprob
