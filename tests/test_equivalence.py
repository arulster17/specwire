import pytest
from transformers import AutoTokenizer

from specwire.baseline import target_only, target_only_cached
from specwire.draft import Drafter
from specwire.orchestrator import generate
from specwire.target import Verifier

PROMPTS = [
    "Once upon a time, in a small village by the sea,",
    "The capital of France is",
    "# Python function that returns the nth Fibonacci number\ndef",
]


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")


@pytest.fixture(scope="module")
def drafter():
    return Drafter("Qwen/Qwen2.5-0.5B")


@pytest.fixture(scope="module")
def verifier():
    return Verifier("Qwen/Qwen2.5-1.5B")


@pytest.mark.parametrize("k", [1, 3, 5])
@pytest.mark.parametrize("prompt", PROMPTS)
def test_matches_target_only(tokenizer, drafter, verifier, prompt, k):
    ids = tokenizer(prompt)["input_ids"]
    eos = tokenizer.eos_token_id
    assert generate(drafter, verifier, ids, 32, k, eos) == target_only(
        verifier.model, ids, 32, eos
    )


def test_verify_rejects_keep_len_past_cache(verifier):
    verifier.verify(0, [1, 2, 3], [4])  # cache now holds 4 tokens
    with pytest.raises(ValueError):
        verifier.verify(5, [6], [7])


def test_sampling_is_reproducible_and_seed_dependent(tokenizer, drafter, verifier):
    ids = tokenizer(PROMPTS[0])["input_ids"]
    runs = [generate(drafter, verifier, ids, 24, 4, temperature=1.0, seed=s) for s in (0, 0, 1)]
    assert runs[0] == runs[1]
    assert runs[0] != runs[2]


@pytest.mark.parametrize("prompt", PROMPTS)
def test_cached_baseline_matches_uncached(tokenizer, verifier, prompt):
    ids = tokenizer(prompt)["input_ids"]
    eos = tokenizer.eos_token_id
    assert target_only_cached(verifier.model, ids, 32, eos) == target_only(
        verifier.model, ids, 32, eos
    )
