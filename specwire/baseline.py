import torch
from transformers import DynamicCache


def target_only(
    model, prompt_ids: list[int], max_new_tokens: int, eos_token_id: int | None = None
) -> list[int]:
    """Greedy-decode with the target alone, one full forward pass per token.

    Deliberately simple: the correctness reference. Time target_only_cached instead.
    """
    generated = []
    with torch.inference_mode():
        while len(generated) < max_new_tokens:
            logits = model(torch.tensor([prompt_ids + generated], device=model.device)).logits
            token = int(logits[0, -1].argmax())
            generated.append(token)
            if token == eos_token_id:
                break
    return generated


def target_only_cached(
    model, prompt_ids: list[int], max_new_tokens: int, eos_token_id: int | None = None
) -> list[int]:
    """Greedy-decode with the target alone using a KV cache: the fair speed baseline."""
    cache = DynamicCache()
    generated = []
    feed = prompt_ids
    with torch.inference_mode():
        while len(generated) < max_new_tokens:
            logits = model(
                torch.tensor([feed], device=model.device), past_key_values=cache, use_cache=True
            ).logits
            token = int(logits[0, -1].argmax())
            generated.append(token)
            if token == eos_token_id:
                break
            feed = [token]
    return generated
