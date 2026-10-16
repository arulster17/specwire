import time


def generate(
    drafter,
    verifier,
    prompt_ids: list[int],
    max_new_tokens: int,
    k: int,
    eos_token_id: int | None = None,
    on_round=None,
) -> list[int]:
    """Greedy speculative decoding; output matches target_only token for token."""
    output = []
    while len(output) < max_new_tokens:
        context = prompt_ids + output
        start = time.perf_counter()
        draft = drafter.propose(context, k)
        draft_s = time.perf_counter() - start
        start = time.perf_counter()
        n, nxt, target_choices = verifier.verify(context, draft)
        verify_s = time.perf_counter() - start
        output += draft[:n] + [nxt]
        if on_round:
            on_round(draft, n, nxt, target_choices, draft_s, verify_s)
        if eos_token_id is not None and eos_token_id in output:
            return output[: output.index(eos_token_id) + 1][:max_new_tokens]
    return output[:max_new_tokens]
