import torch


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


def sample_accept(
    draft: list[int],
    draft_dists: list[tuple[list[int], list[float]]],
    target_probs: torch.Tensor,
    generator: torch.Generator,
) -> tuple[int, int]:
    """Return (num_accepted, next_token) for speculative rejection sampling.

    draft_dists[i] is the sparse (ids, probs) distribution the draft sampled draft[i] from;
    target_probs is (len(draft) + 1, vocab). Output follows target_probs exactly.
    """
    if target_probs.shape[0] != len(draft) + 1:
        raise ValueError(
            f"expected {len(draft) + 1} rows of target probs, got {target_probs.shape[0]}"
        )
    if len(draft_dists) != len(draft):
        raise ValueError(f"expected {len(draft)} draft dists, got {len(draft_dists)}")
    for i, token in enumerate(draft):
        ids, probs = draft_dists[i]
        q = torch.zeros_like(target_probs[i])
        q[torch.tensor(ids, dtype=torch.long)] = torch.tensor(probs, dtype=q.dtype)
        if q[token] <= 0:
            raise ValueError(f"draft token {token} has zero draft probability")
        p = target_probs[i]
        if torch.rand((), generator=generator) < p[token] / q[token]:
            continue
        # Rejected: sample from where the target wants more than the draft offered.
        residual = (p - q).clamp(min=0)
        if residual.sum() <= 0:  # p == q up to float noise
            residual = p
        return i, int(torch.multinomial(residual, 1, generator=generator))
    return len(draft), int(torch.multinomial(target_probs[-1], 1, generator=generator))
