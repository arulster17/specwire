import torch
from transformers import AutoModelForCausalLM

from specwire.verify import greedy_accept


class Verifier:
    """Target model that checks a draft in one forward pass, no KV cache."""

    def __init__(self, model_name: str):
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
        self.model.eval()

    def verify(self, context: list[int], draft: list[int]) -> tuple[int, int, list[int]]:
        """Return (num_accepted, next_token, target_choices) for draft following context."""
        with torch.inference_mode():
            logits = self.model(torch.tensor([context + draft])).logits
        target_choices = logits[0, len(context) - 1 :].argmax(-1).tolist()
        n, nxt = greedy_accept(draft, target_choices)
        return n, nxt, target_choices
