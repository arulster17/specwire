def generate(
    drafter,
    verifier,
    prompt_ids: list[int],
    max_new_tokens: int,
    k: int,
    eos_token_id: int | None = None,
) -> list[int]:
    """Greedy speculative decoding; output matches target_only token for token."""
    output = []
    while len(output) < max_new_tokens:
        context = prompt_ids + output
        draft = drafter.propose(context, k)
        n, nxt = verifier.verify(context, draft)
        output += draft[:n] + [nxt]
        if eos_token_id is not None and eos_token_id in output:
            return output[: output.index(eos_token_id) + 1][:max_new_tokens]
    return output[:max_new_tokens]
