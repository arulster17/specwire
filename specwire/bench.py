"""Benchmark target-only, co-located and split speculative decoding.

    uv run python -m specwire.bench --modes target colocated split --k 1 3 5 --runs 3

Split mode starts both servers locally as subprocesses unless --draft/--target
point at running ones. Every run decodes a fixed number of tokens (no EOS stop),
so runs are the same amount of work.

Columns: tok/s is the median over runs (range = min-max); speedup is vs target-only;
acc/rnd = draft tokens accepted per round; tok/call = tokens per target call;
draft ms/tok and verify ms are model compute per round; net ms = call time minus
compute (gRPC + injected delay); cost = draft ms per token / target-only ms per token;
match = prompts whose greedy output equals target-only.

Second table: tgt GB / draft GB = peak GPU memory allocated on the target's and
the draft's GPU (co-located counts both models on one GPU; target-only isn't
measured because the idle draft is loaded too); $/1M tok = dollars per million
tokens from --price-target (plus --price-draft for split).
"""

import argparse
import json
import os
import socket
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from rich.console import Console
from rich.table import Table
from transformers import AutoTokenizer

from specwire.baseline import target_only_cached, target_only_static
from specwire.draft import Drafter
from specwire.orchestrator import generate
from specwire.rpc import TOKEN_ENV, DraftClient, TargetClient, gpu_memory
from specwire.target import Verifier

PROMPTS = [
    "Once upon a time, in a small village by the sea,",
    "# Python function that returns the nth Fibonacci number\ndef",
    "The three main causes of the French Revolution were",
    "Here is a list of ten fruits: apple, banana,",
]


class Delayed:
    """Worker client that sleeps rtt_ms before each call, to simulate network delay."""

    def __init__(self, worker, rtt_ms: float):
        self.worker = worker
        self.rtt_s = rtt_ms / 1000

    def propose(self, *args):
        time.sleep(self.rtt_s)
        return self.worker.propose(*args)

    def verify(self, *args):
        time.sleep(self.rtt_s)
        return self.worker.verify(*args)

    def __getattr__(self, name):
        return getattr(self.worker, name)


def run_once(mode, drafter, verifier, prompt_ids, max_new_tokens, k, temperature, top_k, seed):
    """Decode every prompt once; return the run's tokens, time, outputs and per-round stats."""
    outputs, rounds = [], []

    def on_round(draft, n, nxt, choices, draft_s, verify_s):
        rounds.append(
            {
                "accepted": n,
                "draft_s": draft_s,
                "verify_s": verify_s,
                # Remote clients report server-side time; in-process, call time is compute time.
                "draft_compute_s": getattr(drafter, "last_compute_s", draft_s),
                "verify_compute_s": getattr(verifier, "last_compute_s", verify_s),
            }
        )

    start = time.perf_counter()
    for i, ids in enumerate(prompt_ids):
        if mode == "target" and verifier.runner:
            outputs.append(target_only_static(verifier.runner, ids, max_new_tokens))
        elif mode == "target":
            outputs.append(target_only_cached(verifier.model, ids, max_new_tokens))
        else:
            outputs.append(
                generate(
                    drafter,
                    verifier,
                    ids,
                    max_new_tokens,
                    k,
                    on_round=on_round,
                    temperature=temperature,
                    top_k=top_k,
                    seed=seed + i,
                )
            )
    seconds = time.perf_counter() - start
    return {
        "tokens": sum(len(o) for o in outputs),
        "seconds": seconds,
        "outputs": outputs,
        "rounds": rounds,
    }


def summarize(config, runs, target_ms_per_token=None, target_outputs=None):
    """Median and spread of tokens/sec across runs, plus per-round averages."""
    tok_s = [r["tokens"] / r["seconds"] for r in runs]
    summary = {
        "tok_s_median": statistics.median(tok_s),
        "tok_s_min": min(tok_s),
        "tok_s_max": max(tok_s),
        "ms_per_token": 1000 * statistics.median(r["seconds"] / r["tokens"] for r in runs),
    }
    rounds = [rd for r in runs for rd in r["rounds"]]
    if rounds:
        k = config["k"]
        mean = lambda key: statistics.fmean(rd[key] for rd in rounds)  # noqa: E731
        summary |= {
            "accepted_per_round": mean("accepted"),
            "tokens_per_target_call": sum(r["tokens"] for r in runs) / len(rounds),
            "draft_ms_per_token": 1000 * mean("draft_compute_s") / k,
            "verify_ms_per_round": 1000 * mean("verify_compute_s"),
            "network_ms_per_round": 1000
            * (mean("draft_s") - mean("draft_compute_s") + mean("verify_s") - mean("verify_compute_s")),
        }
        if target_ms_per_token:
            # The draft/target cost ratio: speculation only pays when this is small.
            summary["draft_target_cost_ratio"] = summary["draft_ms_per_token"] / target_ms_per_token
    if target_outputs is not None:
        outputs = runs[0]["outputs"]
        summary["matches_target_only"] = sum(a == b for a, b in zip(outputs, target_outputs))
        summary["prompts"] = len(outputs)
    return summary


def worker_memory(worker, reset=False):
    """Peak GPU bytes of a worker: asked over RPC for a remote one, read directly in-process."""
    if hasattr(worker, "memory"):
        return worker.memory(reset)
    return gpu_memory(worker.device, reset)


def usd_per_million(tok_s, usd_per_hour):
    return usd_per_hour / (tok_s * 3600) * 1e6


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def spawn_server(role, model, device, dtype, fast):
    """Start a worker server subprocess; return (process, address) once it is listening."""
    port = free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "specwire.rpc", role, "--port", str(port), "--model", model]
        + ["--device", device, "--dtype", dtype]
        + (["--fast"] if fast else []),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",  # Windows would default to cp1252 and choke on progress bars
        errors="replace",
    )
    log = []
    for line in proc.stdout:
        log.append(line)
        if "listening" in line:
            return proc, f"localhost:{port}"
    raise RuntimeError(f"{role} server exited before listening:\n{''.join(log[-20:])}")


def bench(configs, drafter, verifier, prompt_ids, args, console, baseline):
    """Run each config (warmup + runs) and return result records."""
    results = []
    for config in configs:
        worker_d, worker_t = drafter, verifier
        if config["rtt_ms"]:
            worker_d, worker_t = Delayed(drafter, config["rtt_ms"]), Delayed(verifier, config["rtt_ms"])
        run = lambda seed: run_once(  # noqa: E731
            config["mode"],
            worker_d,
            worker_t,
            prompt_ids,
            args.max_new_tokens,
            config["k"],
            args.temperature,
            args.top_k,
            seed,
        )
        split = config["mode"] == "split"
        # In-process, both models share one GPU and one peak; split reads each server.
        worker_memory(verifier, reset=True)
        if split:
            worker_memory(drafter, reset=True)
        for w in range(args.warmup):
            run(args.seed - 1 - w)
        runs = [run(args.seed + 1000 * i) for i in range(args.runs)]
        memory = {"target": worker_memory(verifier)}
        if split:
            memory["draft"] = worker_memory(drafter)
        # Greedy output should match target-only; sampled output can't be compared.
        compare = args.temperature == 0 and config["mode"] != "target"
        summary = summarize(
            config,
            runs,
            baseline.get("ms_per_token"),
            baseline.get("outputs") if compare else None,
        )
        if config["mode"] == "target":
            baseline |= {
                "ms_per_token": summary["ms_per_token"],
                "outputs": runs[0]["outputs"],
                "tok_s": summary["tok_s_median"],
            }
        if baseline.get("tok_s"):
            summary["speedup"] = summary["tok_s_median"] / baseline["tok_s"]
        # Target-only runs with the idle draft loaded too, so its peak would overstate
        # a real target-only GPU; split's target server is the honest number for that.
        if config["mode"] != "target":
            summary["gpu_memory"] = memory
        if args.price_target is not None:
            usd_per_hour = args.price_target + (args.price_draft if split else 0)
            summary["usd_per_hour"] = usd_per_hour
            summary["usd_per_1m_tokens"] = usd_per_million(summary["tok_s_median"], usd_per_hour)
        console.print(f"done: {label(config)}  {summary['tok_s_median']:.1f} tok/s")
        results.append({"config": config, "summary": summary, "runs": runs})
    return results


def label(config):
    if config["mode"] == "target":
        return "target-only"
    rtt = f" +{config['rtt_ms']:g}ms" if config["rtt_ms"] else ""
    return f"{config['mode']} k={config['k']}{rtt}"


def print_table(results, console):
    # Short headers so the table fits 80 columns; see the module docstring for meanings.
    table = Table(box=None)
    table.add_column("config", no_wrap=True)
    for col in ("tok/s", "range", "speedup", "acc/rnd", "tok/call"):
        table.add_column(col)
    for col in ("draft ms/tok", "verify ms", "net ms", "cost", "match"):
        table.add_column(col, justify="right")
    fmt = lambda v, spec=".2f": "-" if v is None else format(v, spec)  # noqa: E731
    for r in results:
        s = r["summary"]
        table.add_row(
            label(r["config"]),
            f"{s['tok_s_median']:.1f}",
            f"{s['tok_s_min']:.1f}-{s['tok_s_max']:.1f}",
            fmt(s.get("speedup")) + ("x" if s.get("speedup") else ""),
            fmt(s.get("accepted_per_round")),
            fmt(s.get("tokens_per_target_call")),
            fmt(s.get("draft_ms_per_token"), ".1f"),
            fmt(s.get("verify_ms_per_round"), ".1f"),
            fmt(s.get("network_ms_per_round") if r["config"]["mode"] == "split" else None, ".1f"),
            fmt(s.get("draft_target_cost_ratio")),
            f"{s['matches_target_only']}/{s['prompts']}" if "matches_target_only" in s else "-",
        )
    console.print(table)

    # Memory and cost go in their own table so the main one doesn't get wider.
    console.print()
    table = Table(box=None)
    table.add_column("config", no_wrap=True)
    for col in ("tgt GB", "draft GB", "$/hour", "$/1M tok"):
        table.add_column(col, justify="right")
    gb = lambda s, worker: s.get("gpu_memory", {}).get(worker, {}).get("peak_allocated")  # noqa: E731
    for r in results:
        s = r["summary"]
        tgt, draft = gb(s, "target"), gb(s, "draft")
        table.add_row(
            label(r["config"]),
            fmt(tgt / 1e9 if tgt is not None else None),
            fmt(draft / 1e9 if draft is not None else None),
            fmt(s.get("usd_per_hour")),
            fmt(s.get("usd_per_1m_tokens")),
        )
    console.print(table)


def main():
    cuda = torch.cuda.is_available()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--modes", nargs="+", default=["target", "colocated", "split"],
                        choices=["target", "colocated", "split"])
    parser.add_argument("--k", nargs="+", type=int, default=[5])
    parser.add_argument("--rtt-ms", nargs="+", type=float, default=[0.0],
                        help="extra delay per RPC call, split mode only")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--draft-model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--target-model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--device", default="cuda" if cuda else "cpu")
    parser.add_argument("--dtype", default="bfloat16" if cuda else "float32",
                        choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--fast", action="store_true",
                        help="static cache + CUDA graphs for draft, target and baseline")
    parser.add_argument("--draft", help="address of a running draft server (split mode)")
    parser.add_argument("--target", help="address of a running target server (split mode)")
    parser.add_argument("--price-target", type=float,
                        help="target GPU price in $/hour, to report $ per 1M tokens")
    parser.add_argument("--price-draft", type=float,
                        help="draft GPU price in $/hour, added for split mode")
    parser.add_argument("--out", help="results JSON path (default: results/bench-<time>.json)")
    args = parser.parse_args()
    if args.price_target is not None and "split" in args.modes and args.price_draft is None:
        parser.error("--price-target with split mode needs --price-draft too")

    console = Console(highlight=False)
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    prompt_ids = [tokenizer(p)["input_ids"] for p in PROMPTS]
    dtype = getattr(torch, args.dtype)
    inproc = [m for m in ("target", "colocated") if m in args.modes]
    configs = [{"mode": "target", "k": None, "rtt_ms": 0.0}] if "target" in inproc else []
    if "colocated" in inproc:
        configs += [{"mode": "colocated", "k": k, "rtt_ms": 0.0} for k in args.k]
    results, baseline = [], {}

    if inproc:
        fast = " fast" if args.fast else ""
        console.print(f"loading models on {args.device} ({args.dtype}{fast})")
        drafter = Drafter(args.draft_model, args.device, dtype, args.fast)
        verifier = Verifier(args.target_model, args.device, dtype, args.fast)
        results += bench(configs, drafter, verifier, prompt_ids, args, console, baseline)
        del drafter, verifier  # free the GPU before split mode loads its own copies
        if cuda:
            torch.cuda.empty_cache()

    if "split" in args.modes:
        procs = []
        try:
            draft_addr, target_addr = args.draft, args.target
            if not draft_addr:
                console.print("starting draft server")
                proc, draft_addr = spawn_server(
                    "draft", args.draft_model, args.device, args.dtype, args.fast
                )
                procs.append(proc)
            if not target_addr:
                console.print("starting target server")
                proc, target_addr = spawn_server(
                    "target", args.target_model, args.device, args.dtype, args.fast
                )
                procs.append(proc)
            configs = [{"mode": "split", "k": k, "rtt_ms": r} for k in args.k for r in args.rtt_ms]
            # Spawned servers inherit the same env, so a set token covers both sides.
            token = os.environ.get(TOKEN_ENV)
            results += bench(
                configs, DraftClient(draft_addr, token), TargetClient(target_addr, token),
                prompt_ids, args, console, baseline,
            )
        finally:
            for proc in procs:
                proc.terminate()

    print_table(results, console)
    out = Path(args.out or f"results/bench-{datetime.now():%Y%m%d-%H%M%S}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {k: v for k, v in vars(args).items() if k != "out"}
    meta["gpu"] = torch.cuda.get_device_name(0) if cuda and args.device.startswith("cuda") else None
    meta["prompts"] = PROMPTS
    out.write_text(json.dumps({"args": meta, "results": results}, indent=1))
    console.print(f"wrote {out}")


if __name__ == "__main__":
    main()
