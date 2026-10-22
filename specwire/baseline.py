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


def target_only_static(
    runner, prompt_ids: list[int], max_new_tokens: int, eos_token_id: int | None = None
) -> list[int]:
    """Greedy-decode with the target alone on a StaticRunner: the fair baseline for
    fast mode. Tokens stay on the device, so it waits for the GPU once per sequence;
    an EOS is cut afterwards, which gives the same output as stopping there."""
    runner.rollback(0)
    feed = torch.tensor(prompt_ids, device=runner.device)
    generated = []
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            feed = runner(feed)[-1].argmax().view(1)
            generated.append(feed)
    generated = torch.cat(generated).tolist() if generated else []
    if eos_token_id in generated:
        generated = generated[: generated.index(eos_token_id) + 1]
    return generated
