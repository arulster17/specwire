import time

import pytest
import torch

from specwire.bench import Delayed, summarize, usd_per_million, worker_memory


def fake_run(tokens, seconds, accepted, draft_s=0.02, compute_s=0.01, outputs=()):
    rounds = [
        {
            "accepted": n,
            "draft_s": draft_s,
            "verify_s": draft_s,
            "draft_compute_s": compute_s,
            "verify_compute_s": compute_s,
        }
        for n in accepted
    ]
    return {"tokens": tokens, "seconds": seconds, "outputs": list(outputs), "rounds": rounds}


def test_summarize_rates_and_network_time():
    runs = [fake_run(8, 1.0, [3, 1]), fake_run(8, 2.0, [3, 1]), fake_run(8, 4.0, [3, 1])]
    s = summarize({"mode": "split", "k": 4}, runs, target_ms_per_token=5.0)
    assert (s["tok_s_median"], s["tok_s_min"], s["tok_s_max"]) == (4.0, 2.0, 8.0)
    assert s["accepted_per_round"] == 2.0
    assert s["tokens_per_target_call"] == 4.0  # 24 tokens over 6 rounds
    assert s["draft_ms_per_token"] == pytest.approx(2.5)  # 10 ms per round / k=4
    assert s["network_ms_per_round"] == pytest.approx(20.0)  # (20 - 10) ms on each call
    assert s["draft_target_cost_ratio"] == pytest.approx(0.5)


def test_summarize_counts_matching_outputs():
    runs = [fake_run(4, 1.0, [1], outputs=[[1, 2], [3, 4]])]
    s = summarize({"mode": "colocated", "k": 1}, runs, target_outputs=[[1, 2], [3, 5]])
    assert (s["matches_target_only"], s["prompts"]) == (1, 2)


def test_delayed_sleeps_and_forwards():
    class Worker:
        last_compute_s = 0.25

        def propose(self, *args):
            return args

    delayed = Delayed(Worker(), rtt_ms=50)
    start = time.perf_counter()
    assert delayed.propose(1, 2) == (1, 2)
    assert time.perf_counter() - start >= 0.05
    assert delayed.last_compute_s == 0.25


def test_usd_per_million():
    # $3.60/hour at 100 tok/s is 360,000 tokens per $3.60, so $10 per 1M tokens.
    assert usd_per_million(100.0, 3.6) == pytest.approx(10.0)


def test_worker_memory_prefers_rpc_and_falls_back_to_device():
    class Remote:
        def memory(self, reset=False):
            return {"peak_allocated": 7, "peak_reserved": 9, "reset": reset}

    class Local:
        device = torch.device("cpu")

    assert worker_memory(Remote(), reset=True)["reset"] is True
    assert worker_memory(Delayed(Remote(), rtt_ms=0))["peak_allocated"] == 7
    assert worker_memory(Local()) == {"peak_allocated": 0, "peak_reserved": 0}
    assert worker_memory(Delayed(Local(), rtt_ms=0)) == {"peak_allocated": 0, "peak_reserved": 0}
