import torch
from transformers import AutoModelForCausalLM, DynamicCache

from specwire.kv import reuse_cache
from specwire.verify import greedy_accept, sample_accept


class Verifier:
    """Target model that checks a draft in one forward pass, with a KV cache."""

    def __init__(self, model_name: str, device: str = "cpu", dtype: torch.dtype = torch.float32):
        self.device = torch.device(device)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(self.device)
        self.model.eval()
        self.cache = DynamicCache()

    def verify(
        self,
        keep_len: int,
        new_tokens: list[int],
        draft: list[int],
        draft_dists: list[tuple[list[int], list[float]]] = (),
        temperature: float = 0.0,
        seed: int = 0,
    ) -> tuple[int, int, list[int]]:
        """Return (num_accepted, next_token, target_choices) after keep_len cached + new_tokens.

        Greedy at temperature 0; otherwise rejection sampling against draft_dists.
        """
        self.cache = reuse_cache(self.cache, keep_len, new_tokens)
        with torch.inference_mode():
            logits = self.model(
                torch.tensor([new_tokens + draft], device=self.device),
                past_key_values=self.cache,
                use_cache=True,
            ).logits[0, len(new_tokens) - 1 :].float()
        target_choices = logits.argmax(-1).tolist()
        if temperature == 0:
            n, nxt = greedy_accept(draft, target_choices)
        else:
            generator = torch.Generator().manual_seed(seed)
            probs = (logits / temperature).softmax(-1).cpu()  # sample on CPU, as in draft
            n, nxt = sample_accept(draft, list(draft_dists), probs, generator)
        return n, nxt, target_choices
