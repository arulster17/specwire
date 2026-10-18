import torch
from transformers import AutoModelForCausalLM, DynamicCache

from specwire.kv import reuse_cache


class Drafter:
    """Draft model with a KV cache; greedy at temperature 0, else top-k sampling."""

    def __init__(self, model_name: str):
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
        self.model.eval()
        self.cache = DynamicCache()

    def propose(
        self,
        keep_len: int,
        new_tokens: list[int],
        k: int,
        temperature: float = 0.0,
        top_k: int = 20,
        seed: int = 0,
    ) -> tuple[list[int], list[tuple[list[int], list[float]]]]:
        """Return k draft tokens following keep_len cached tokens + new_tokens, and the
        sparse (ids, probs) distribution each was drawn from (one-hot when greedy)."""
        self.cache = reuse_cache(self.cache, keep_len, new_tokens)
        generator = torch.Generator().manual_seed(seed)
        drafted, dists = [], []
        feed = new_tokens
        with torch.inference_mode():
            for _ in range(k):
                logits = self.model(
                    torch.tensor([feed]), past_key_values=self.cache, use_cache=True
                ).logits[0, -1]
                if temperature == 0:
                    token = int(logits.argmax())
                    dists.append(([token], [1.0]))
                else:
                    top = (logits / temperature).topk(min(top_k, logits.shape[-1]))
                    probs = top.values.softmax(-1)
                    token = int(top.indices[torch.multinomial(probs, 1, generator=generator)])
                    dists.append((top.indices.tolist(), probs.tolist()))
                drafted.append(token)
                feed = [token]
        return drafted, dists
