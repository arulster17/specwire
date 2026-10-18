import pytest
import torch

from specwire.orchestrator import round_seed
from specwire.verify import greedy_accept, sample_accept


def test_all_guesses_accepted():
    assert greedy_accept([5, 9, 2, 7], [5, 9, 2, 7, 8]) == (4, 8)


def test_first_guess_wrong():
    assert greedy_accept([5, 9], [4, 9, 1]) == (0, 4)


def test_mismatch_in_middle():
    assert greedy_accept([5, 9, 2, 7], [5, 9, 3, 1, 8]) == (2, 3)


def test_empty_draft():
    assert greedy_accept([], [6]) == (0, 6)


def test_wrong_length_raises():
    with pytest.raises(ValueError):
        greedy_accept([5, 9], [5, 9])


# Target distribution over a 4-token vocab, with mass on tokens the draft below never offers.
P = torch.tensor([0.5, 0.3, 0.15, 0.05])


def emitted_freqs(draft_q: dict[int, float], target_rows: torch.Tensor, trials=40_000):
    """Frequency of the first emitted token when draft guesses one token from draft_q."""
    g = torch.Generator().manual_seed(0)
    ids, probs = list(draft_q), list(draft_q.values())
    counts = torch.zeros(target_rows.shape[1])
    for _ in range(trials):
        guess = ids[int(torch.multinomial(torch.tensor(probs), 1, generator=g))]
        n, nxt = sample_accept([guess], [(ids, probs)], target_rows, g)
        counts[guess if n == 1 else nxt] += 1
    return counts / trials


@pytest.mark.parametrize(
    "draft_q",
    [
        {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25},  # full support, wrong shape
        {0: 0.2, 1: 0.8},  # top-2 truncated: tokens 2 and 3 only reachable via rejection
        {1: 1.0},  # greedy draft that disagrees with the target's favorite
    ],
)
def test_sampled_output_follows_target(draft_q):
    rows = torch.stack([P, torch.full((4,), 0.25)])
    assert torch.allclose(emitted_freqs(draft_q, rows), P, atol=0.01)


def test_bonus_token_follows_target():
    g = torch.Generator().manual_seed(0)
    bonus = torch.tensor([0.1, 0.2, 0.3, 0.4])
    counts = torch.zeros(4)
    for _ in range(40_000):
        # Draft matches the target exactly at spot 1, so it is always accepted.
        n, nxt = sample_accept([2], [([2], [1.0])], torch.stack([torch.eye(4)[2], bonus]), g)
        assert n == 1
        counts[nxt] += 1
    assert torch.allclose(counts / 40_000, bonus, atol=0.01)


def test_draft_equal_to_target_always_accepted():
    g = torch.Generator().manual_seed(0)
    rows = P.repeat(4, 1)
    dist = ([0, 1, 2, 3], P.tolist())
    for _ in range(200):
        draft = torch.multinomial(P, 3, replacement=True, generator=g).tolist()
        assert sample_accept(draft, [dist] * 3, rows, g)[0] == 3


def test_sample_wrong_lengths_raise():
    g = torch.Generator()
    with pytest.raises(ValueError):
        sample_accept([0, 1], [([0], [1.0])] * 2, P.repeat(2, 1), g)
    with pytest.raises(ValueError):
        sample_accept([0, 1], [([0], [1.0])], P.repeat(3, 1), g)


def test_sample_rejects_token_draft_could_not_draw():
    with pytest.raises(ValueError):
        sample_accept([3], [([0, 1], [0.5, 0.5])], P.repeat(2, 1), torch.Generator())


def test_round_seeds_differ_by_role_round_and_run():
    seeds = {round_seed(s, r, role) for s in range(3) for r in range(3) for role in ("draft", "target")}
    assert len(seeds) == 18
    assert round_seed(7, 2, "draft") == round_seed(7, 2, "draft")
