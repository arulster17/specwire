import grpc
import pytest
from transformers import AutoTokenizer

from specwire.baseline import target_only
from specwire.draft import Drafter
from specwire.orchestrator import generate
from specwire.proto import specwire_pb2
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
def drafter():
    return Drafter("Qwen/Qwen2.5-0.5B")


@pytest.fixture(scope="module")
def draft_client(drafter):
    # localhost, not [::], so macOS doesn't raise a firewall prompt.
    server, port = start_draft_server(drafter, "localhost:0")
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


def test_propose_rejects_k_zero(draft_client):
    with pytest.raises(grpc.RpcError) as exc:
        draft_client.stub.Propose(specwire_pb2.ProposeRequest(new_tokens=[1, 2, 3], k=0))
    assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_verify_rejects_lost_cache(target_client):
    target_client.verify(0, [1, 2, 3], [4])  # target now caches 4 tokens
    with pytest.raises(grpc.RpcError) as exc:
        target_client.verify(5, [6], [7])
    assert exc.value.code() == grpc.StatusCode.FAILED_PRECONDITION


def test_rpc_sampling_matches_in_process(tokenizer, drafter, verifier, draft_client, target_client):
    ids = tokenizer(PROMPTS[0])["input_ids"]
    sample = dict(temperature=0.8, top_k=20, seed=3)
    remote = generate(draft_client, target_client, ids, 24, 4, **sample)
    local = generate(drafter, verifier, ids, 24, 4, **sample)
    assert remote == local


@pytest.mark.parametrize(
    "dists",
    [
        [],  # missing
        [specwire_pb2.DraftDist(ids=[5, 6], probs=[1.0])],  # ids/probs length mismatch
        [specwire_pb2.DraftDist(ids=[5], probs=[1.0])],  # draft token 4 not in its dist
    ],
)
def test_verify_rejects_bad_draft_dists(target_client, dists):
    request = specwire_pb2.VerifyRequest(
        new_tokens=[1, 2, 3], draft=[4], draft_dists=dists, temperature=1.0
    )
    with pytest.raises(grpc.RpcError) as exc:
        target_client.stub.Verify(request)
    assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT
