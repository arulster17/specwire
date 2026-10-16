import pytest
from transformers import AutoTokenizer

from specwire.baseline import target_only
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
