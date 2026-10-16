def greedy_accept(draft: list[int], target_choices: list[int]) -> tuple[int, int]:
    """Return (num_accepted, next_token) for greedy speculative verification."""
    if len(target_choices) != len(draft) + 1:
        raise ValueError(
            f"expected {len(draft) + 1} target choices, got {len(target_choices)}"
        )
    for i, token in enumerate(draft):
        if token != target_choices[i]:
            return i, target_choices[i]
    return len(draft), target_choices[-1]
