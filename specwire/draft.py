import torch
from transformers import AutoModelForCausalLM, DynamicCache

from specwire.kv import reuse_cache


class Drafter:
    """Greedy draft model with a KV cache."""

    def __init__(self, model_name: str):
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
        self.model.eval()
        self.cache = DynamicCache()

    def propose(self, keep_len: int, new_tokens: list[int], k: int) -> list[int]:
        """Return k greedy draft tokens following keep_len cached tokens + new_tokens."""
        self.cache = reuse_cache(self.cache, keep_len, new_tokens)
        drafted = []
        feed = new_tokens
        with torch.inference_mode():
            for _ in range(k):
                logits = self.model(
                    torch.tensor([feed]), past_key_values=self.cache, use_cache=True
                ).logits
                drafted.append(int(logits[0, -1].argmax()))
                feed = drafted[-1:]
        return drafted
