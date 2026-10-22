"""Fast mode (static cache + CUDA graphs) must give the same tokens as the simple path.

Runs on CPU (static cache, no graphs) and, when available, on CUDA in float32 (graphs).
"""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from specwire.baseline import target_only, target_only_static
from specwire.draft import Drafter
from specwire.fast import StaticRunner
from specwire.orchestrator import generate
from specwire.rpc import DraftClient, TargetClient, start_draft_server, start_target_server
from specwire.target import Verifier

PROMPTS = [
    "Once upon a time, in a small village by the sea,",
    "The capital of France is",
    "# Python function that returns the nth Fibonacci number\ndef",
]
DEVICES = [
    "cpu",
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    ),
]


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")


@pytest.fixture(scope="module", params=DEVICES)
def workers(request):
    drafter = Drafter("Qwen/Qwen2.5-0.5B", request.param, fast=True)
    verifier = Verifier("Qwen/Qwen2.5-1.5B", request.param, fast=True)
    yield drafter, verifier
    del drafter, verifier
    if request.param == "cuda":
        torch.cuda.empty_cache()


@pytest.mark.parametrize("k", [1, 3, 5])
@pytest.mark.parametrize("prompt", PROMPTS)
def test_fast_matches_target_only(tokenizer, workers, prompt, k):
    drafter, verifier = workers
    ids = tokenizer(prompt)["input_ids"]
    eos = tokenizer.eos_token_id
    assert generate(drafter, verifier, ids, 32, k, eos) == target_only(
        verifier.model, ids, 32, eos
    )


def test_fast_over_rpc_matches_target_only(tokenizer, workers):
    # The servers check requests against the worker's cache, which in fast mode is the runner.
    drafter, verifier = workers
    draft_server, draft_port = start_draft_server(drafter, "localhost:0")
    target_server, target_port = start_target_server(verifier, "localhost:0")
    try:
        ids = tokenizer(PROMPTS[0])["input_ids"]
        eos = tokenizer.eos_token_id
        out = generate(
            DraftClient(f"localhost:{draft_port}"), TargetClient(f"localhost:{target_port}"),
            ids, 32, 3, eos,
        )
    finally:
        draft_server.stop(0)
        target_server.stop(0)
    assert out == target_only(verifier.model, ids, 32, eos)


@pytest.mark.parametrize("prompt", PROMPTS)
def test_static_baseline_matches_target_only(tokenizer, workers, prompt):
    _, verifier = workers
    ids = tokenizer(prompt)["input_ids"]
    eos = tokenizer.eos_token_id
    assert target_only_static(verifier.runner, ids, 32, eos) == target_only(
        verifier.model, ids, 32, eos
    )


def test_fast_sampling_matches_eager(tokenizer, workers):
    # Same seeds, same distributions: sampled output must match the eager path exactly
    # in float32 (both paths sample on CPU from the same logits up to float error).
    drafter, verifier = workers
    ids = tokenizer(PROMPTS[0])["input_ids"]
    fast = generate(drafter, verifier, ids, 24, 4, temperature=1.0, seed=3)
    drafter.runner = verifier.runner = None
    try:
        eager = generate(drafter, verifier, ids, 24, 4, temperature=1.0, seed=3)
    finally:
        drafter.runner, verifier.runner = StaticRunner(drafter.model), StaticRunner(verifier.model)
    assert fast == eager


def test_rollback_past_cache_raises(workers):
    _, verifier = workers
    verifier.verify(0, [1, 2, 3], [4])  # cache now holds 4 tokens
    with pytest.raises(ValueError):
        verifier.verify(5, [6], [7])


def test_runner_rejects_overflow():
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
    runner = StaticRunner(model, max_len=8)
    with torch.inference_mode():
        runner(torch.arange(6))
        with pytest.raises(ValueError):
            runner(torch.arange(3))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_graph_logits_match_eager_after_rollback():
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B").cuda().eval()
    graphed, eager = StaticRunner(model), StaticRunner(model, graph_max_len=0)
    prompt = torch.arange(100, 120, device="cuda")
    with torch.inference_mode():
        for runner in (graphed, eager):
            runner(prompt)
        # Write 4 tokens, roll 3 back, then continue: stale slots must not leak in.
        for runner in (graphed, eager):
            runner(torch.tensor([5, 6, 7, 8], device="cuda"))
            runner.rollback(21)
        steps = [torch.tensor([9], device="cuda"), torch.tensor([10, 11, 12], device="cuda")]
        for tokens in steps:
            a, b = graphed(tokens).clone(), eager(tokens)
            torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)
