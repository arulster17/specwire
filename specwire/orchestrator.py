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
    draft_cached = target_cached = 0
    while len(output) < max_new_tokens:
        context = prompt_ids + output
        L = len(context)
        start = time.perf_counter()
        draft = drafter.propose(draft_cached, context[draft_cached:], k)
        draft_s = time.perf_counter() - start
        start = time.perf_counter()
        n, nxt, target_choices = verifier.verify(target_cached, context[target_cached:], draft)
        verify_s = time.perf_counter() - start
        # Keep what each worker processed, up to the accepted prefix context + draft[:n].
        # The draft never processes its last guess.
        draft_cached = min(L + k - 1, L + n)
        target_cached = L + n
        output += draft[:n] + [nxt]
        if on_round:
            on_round(draft, n, nxt, target_choices, draft_s, verify_s)
        if eos_token_id is not None and eos_token_id in output:
            return output[: output.index(eos_token_id) + 1][:max_new_tokens]
    return output[:max_new_tokens]
