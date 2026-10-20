import hashlib
import time


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
) -> list[int]:
    """Speculative decoding. At temperature 0 output matches target_only token for token;
    otherwise it is distributed as sampling the target alone at that temperature.
    draft_temperature only affects speed and defaults to temperature."""
    if draft_temperature is None or temperature == 0:
        draft_temperature = temperature
    output = []
    draft_cached = target_cached = 0
    round_index = 0
    while len(output) < max_new_tokens:
        context = prompt_ids + output
        L = len(context)
        # propose/verify return Python ints, which waits for any GPU work to finish,
        # so these timings are complete without an explicit synchronize.
        start = time.perf_counter()
        draft, dists = drafter.propose(
            draft_cached,
            context[draft_cached:],
            k,
            draft_temperature,
            top_k,
            round_seed(seed, round_index, "draft"),
        )
        draft_s = time.perf_counter() - start
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
        # The draft never processes its last guess.
        draft_cached = min(L + k - 1, L + n)
        target_cached = L + n
        output += draft[:n] + [nxt]
        round_index += 1
        if on_round:
            on_round(draft, n, nxt, target_choices, draft_s, verify_s)
        if eos_token_id is not None and eos_token_id in output:
            return output[: output.index(eos_token_id) + 1][:max_new_tokens]
    return output[:max_new_tokens]
