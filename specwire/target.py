import torch
from transformers import AutoModelForCausalLM, DynamicCache

from specwire.kv import reuse_cache
from specwire.verify import greedy_accept


class Verifier:
    """Target model that checks a draft in one forward pass, with a KV cache."""

    def __init__(self, model_name: str):
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
        self.model.eval()
        self.cache = DynamicCache()

    def verify(
        self, keep_len: int, new_tokens: list[int], draft: list[int]
    ) -> tuple[int, int, list[int]]:
        """Return (num_accepted, next_token, target_choices) after keep_len cached + new_tokens."""
        self.cache = reuse_cache(self.cache, keep_len, new_tokens)
        with torch.inference_mode():
            logits = self.model(
                torch.tensor([new_tokens + draft]), past_key_values=self.cache, use_cache=True
            ).logits
        target_choices = logits[0, len(new_tokens) - 1 :].argmax(-1).tolist()
        n, nxt = greedy_accept(draft, target_choices)
        return n, nxt, target_choices
