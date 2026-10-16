import pytest
from transformers import AutoTokenizer

from specwire.baseline import target_only
from specwire.draft import Drafter
from specwire.orchestrator import generate
from specwire.rpc import DraftClient, TargetClient, start_draft_server, start_target_server
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
def verifier():
    return Verifier("Qwen/Qwen2.5-1.5B")


@pytest.fixture(scope="module")
def draft_client():
    # localhost, not [::], so macOS doesn't raise a firewall prompt.
    server, port = start_draft_server(Drafter("Qwen/Qwen2.5-0.5B"), "localhost:0")
    yield DraftClient(f"localhost:{port}")
    server.stop(0)


@pytest.fixture(scope="module")
def target_client(verifier):
    server, port = start_target_server(verifier, "localhost:0")
    yield TargetClient(f"localhost:{port}")
    server.stop(0)


@pytest.mark.parametrize("prompt", PROMPTS)
def test_rpc_matches_target_only(tokenizer, verifier, draft_client, target_client, prompt):
    ids = tokenizer(prompt)["input_ids"]
    eos = tokenizer.eos_token_id
    assert generate(draft_client, target_client, ids, 32, 5, eos) == target_only(
        verifier.model, ids, 32, eos
    )
