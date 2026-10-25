import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext


def round_seed(seed: int, round_index: int, role: str) -> int:
    """Seed for one worker in one round; distinct per role so the draft's sampling and
    the target's accept tests never share random numbers."""
    digest = hashlib.blake2b(f"{seed}:{round_index}:{role}".encode(), digest_size=8).digest()
    return int.from_bytes(digest) >> 1  # 63 bits fits any seed field


def generate(
    drafter,
    verifier,
    prompt_ids: list[int],
    max_new_tokens: int,
    k: int,
    eos_token_id: int | None = None,
    on_round=None,
    temperature: float = 0.0,
    draft_temperature: float | None = None,
    top_k: int = 20,
    seed: int = 0,
    overlap: bool = False,
) -> list[int]:
    """Speculative decoding. At temperature 0 output matches target_only token for token;
    otherwise it is distributed as sampling the target alone at that temperature.
    draft_temperature only affects speed and defaults to temperature.

    overlap=True drafts the next round while the target verifies this one, betting that
    the target accepts all k tokens and that its next token is the one the draft would
    pick. When the bet wins the next round skips drafting; when it loses the extra draft
    is thrown away, so the output is the same either way. It needs the draft and target
    on separate devices to save time.

    on_round(draft, n, nxt, target_choices, draft_s, verify_s, reused) is called after
    each round: draft_s is time spent waiting on the draft, verify_s on the target, and
    reused says the round's draft came from the previous round's bet.
    """
    if draft_temperature is None or temperature == 0:
        draft_temperature = temperature
    # One thread runs the draft ahead while this one waits on the target.
    with ThreadPoolExecutor(1) if overlap else nullcontext() as pool:
        output = []
        draft_cached = target_cached = 0
        round_index = 0
        bet = None  # (draft, dists) for this round, drafted during the last verify
        while len(output) < max_new_tokens:
            context = prompt_ids + output
            L = len(context)
            # propose/verify return Python ints, which waits for any GPU work to finish,
            # so these timings are complete without an explicit synchronize.
            start = time.perf_counter()
            reused = bet is not None
            if reused:
                draft, dists = bet
            else:
                draft, dists = drafter.propose(
                    draft_cached,
                    context[draft_cached:],
                    k,
                    draft_temperature,
                    top_k,
                    round_seed(seed, round_index, "draft"),
                )
            draft_s = time.perf_counter() - start
            ahead = None
            if pool and len(output) + k + 1 < max_new_tokens:
                # The draft has processed context + draft[:-1]. Feed it the last guess and
                # draft k + 1 more: a guess at the target's next token, then the next draft.
                ahead = pool.submit(
                    drafter.propose,
                    L + k - 1,
                    draft[-1:],
                    k + 1,
                    draft_temperature,
                    top_k,
                    round_seed(seed, round_index + 1, "draft"),
                )
            start = time.perf_counter()
            n, nxt, target_choices = verifier.verify(
                target_cached,
                context[target_cached:],
                draft,
                dists,
                temperature,
                round_seed(seed, round_index, "target"),
            )
            verify_s = time.perf_counter() - start
            # Keep what each worker processed, up to the accepted prefix context + draft[:n].
            # The draft never processes its last guess, unless it drafted ahead.
            draft_cached = min(L + k - 1, L + n)
            target_cached = L + n
            bet = None
            if ahead:
                start = time.perf_counter()
                guess, guess_dists = ahead.result()
                draft_s += time.perf_counter() - start
                draft_cached = L + n
                if n == k and nxt == guess[0]:
                    # The draft already holds context + draft + guess[:-1], which is the
                    # next round's context + its draft[:-1], as if it had proposed it.
                    bet = guess[1:], guess_dists[1:]
            output += draft[:n] + [nxt]
            round_index += 1
            if on_round:
                on_round(draft, n, nxt, target_choices, draft_s, verify_s, reused)
            if eos_token_id is not None and eos_token_id in output:
                return output[: output.index(eos_token_id) + 1][:max_new_tokens]
        return output[:max_new_tokens]
