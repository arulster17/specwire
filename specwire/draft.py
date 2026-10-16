import torch
from transformers import AutoModelForCausalLM


class Drafter:
    """Greedy draft model with no KV cache."""

    def __init__(self, model_name: str):
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
        self.model.eval()

    def propose(self, context: list[int], k: int) -> list[int]:
        """Return k greedy draft tokens following context."""
        drafted = []
        with torch.inference_mode():
            for _ in range(k):
                logits = self.model(torch.tensor([context + drafted])).logits
                drafted.append(int(logits[0, -1].argmax()))
        return drafted
