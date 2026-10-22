import torch
from transformers import AutoModelForCausalLM, DynamicCache

from specwire.fast import StaticRunner
from specwire.kv import reuse_cache


class Drafter:
    """Draft model with a KV cache; greedy at temperature 0, else top-k sampling.

    fast=True uses a static cache with CUDA graph replay (see StaticRunner).
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        fast: bool = False,
    ):
        self.device = torch.device(device)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(self.device)
        self.model.eval()
        self.cache = DynamicCache()
        self.runner = StaticRunner(self.model) if fast else None

    @property
    def cached_len(self) -> int:
        """Tokens currently held in the KV cache."""
        return self.runner.length if self.runner else self.cache.get_seq_length()

    @property
    def max_len(self) -> int:
        """Longest sequence the worker can process."""
        if self.runner:
            return self.runner.max_len
        return self.model.config.max_position_embeddings

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
        if self.runner:
            self.runner.rollback(keep_len)
        else:
            self.cache = reuse_cache(self.cache, keep_len, new_tokens)
        generator = torch.Generator().manual_seed(seed)
        drafted, dists = [], []
        feed = torch.tensor(new_tokens, device=self.device)
        with torch.inference_mode():
            for _ in range(k):
                logits = self._forward(feed)[-1]
                if temperature == 0:
                    # Stays on the device, so greedy drafting waits for the GPU once per round.
                    token = logits.argmax().view(1)
                else:
                    top = (logits.float() / temperature).topk(min(top_k, logits.shape[-1]))
                    # Sample on CPU so a seed gives the same draws on any device.
                    probs, ids = top.values.softmax(-1).cpu(), top.indices.cpu()
                    token = ids[torch.multinomial(probs, 1, generator=generator)]
                    dists.append((ids.tolist(), probs.tolist()))
                    token = token.to(self.device)
                drafted.append(token)
                feed = token
        drafted = torch.cat(drafted).tolist()
        if temperature == 0:
            dists = [([token], [1.0]) for token in drafted]
        return drafted, dists

    def _forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.runner:
            return self.runner(tokens)
        return self.model(tokens[None], past_key_values=self.cache, use_cache=True).logits[0]
