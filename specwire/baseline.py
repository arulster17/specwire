import torch


def target_only(
    model, prompt_ids: list[int], max_new_tokens: int, eos_token_id: int | None = None
) -> list[int]:
    """Greedy-decode with the target alone, one full forward pass per token."""
    generated = []
    with torch.inference_mode():
        while len(generated) < max_new_tokens:
            logits = model(torch.tensor([prompt_ids + generated])).logits
            token = int(logits[0, -1].argmax())
            generated.append(token)
            if token == eos_token_id:
                break
    return generated
