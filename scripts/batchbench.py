"""Predict batched speculative decoding from decode-step timings, before building it.

    python scripts/batchbench.py measure --model meta-llama/Llama-3.1-8B --tokens 1 2 4 6 9 \
        --out results/batch-a100-8b.json
    python scripts/batchbench.py measure --model meta-llama/Llama-3.2-1B --tokens 1 \
        --out results/batch-a100-1b.json
    python scripts/batchbench.py predict --target results/batch-a100-8b.json \
        --draft results/batch-a100-1b.json --split-draft results/batch-4090-1b.json

`measure` times one forward step of B sequences x n new tokens on a static cache with
CUDA graph replay, the way fast mode runs, for every batch size and token count.
`predict` combines those times with the acceptance rates and network time measured in
the single-request cloud runs to estimate throughput per setup and batch size N:

- target alone:   N tokens per step of T(N, 1)
- co-located:     N(a+1) tokens per round of k D_target_gpu(N) + T(N, k+1)
- split:          N(a+1) per round of k D_draft_gpu(N) + T(N, k+1) + network
- split, 2 groups: the users split into two halves that alternate, so the target
  verifies one half while the draft GPU drafts the other. Each period serves both
  halves and lasts max(2 T(N/2, k+1), T(N/2, k+1) + k D(N/2) + network).

a is the mean number of accepted draft tokens per round at draft length k. The model
assumes every sequence in a batch moves in lockstep, ignoring the cost of ragged
acceptance, so it flatters the speculative setups.
"""

import argparse
import json
import statistics
from pathlib import Path

RESULTS = Path("results")


def measure(args):
    import torch
    from transformers import AutoModelForCausalLM, StaticCache

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()
    ctx, max_n = args.context, max(args.tokens)
    out = {"model": args.model, "gpu": torch.cuda.get_device_name(), "context": ctx,
           "replays": args.replays, "ms": {}}

    def forward(cache, ids, positions):
        return model(ids, past_key_values=cache, cache_position=positions, use_cache=True).logits

    def rewind(cache):
        # StaticLayer writes at its own on-device counter (see specwire/fast.py), which
        # every step advances; put it back so each replay writes the same slots.
        for layer in cache.layers:
            layer.cumulative_length.fill_(ctx)

    with torch.inference_mode():
        for batch in args.batches:
            cache = StaticCache(config=model.config, max_cache_len=ctx + max_n)
            prompt = torch.randint(100, 1000, (batch, ctx), device="cuda")
            forward(cache, prompt, torch.arange(ctx, device="cuda"))
            pool = None
            for n in args.tokens:
                rewind(cache)
                ids =torch.randint(100, 1000, (batch, n), device="cuda")
                positions = torch.arange(ctx, ctx + n, device="cuda")
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(2):
                        forward(cache, ids, positions)
                        rewind(cache)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool):
                    forward(cache, ids, positions)
                pool = graph.pool()
                times = []
                for _ in range(args.replays):
                    rewind(cache)
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    graph.replay()
                    end.record()
                    end.synchronize()
                    times.append(start.elapsed_time(end))
                ms = statistics.median(times)
                out["ms"].setdefault(str(batch), {})[str(n)] = ms
                print(f"B={batch:<4} n={n:<2} {ms:7.2f} ms", flush=True)
                del graph
            del cache, pool
            torch.cuda.empty_cache()
    Path(args.out).write_text(json.dumps(out, indent=1))


def accepted_per_round():
    """Mean accepted draft tokens per round at each k, from the single-request A100 runs."""
    results = json.loads((RESULTS / "p3-a100-baselines.json").read_text())["results"]
    return {r["config"]["k"]: r["summary"]["accepted_per_round"]
            for r in results if r["config"]["mode"] == "colocated"}


def network_ms_per_round():
    results = json.loads((RESULTS / "p3-split-4090.json").read_text())["results"]
    return statistics.median(r["summary"]["network_ms_per_round"] for r in results)


def predict(args):
    target, draft, split_draft = (json.loads(Path(p).read_text())["ms"]
                                  for p in (args.target, args.draft, args.split_draft))
    accepted, net = accepted_per_round(), network_ms_per_round()

    def T(b, n):
        return target[str(b)][str(n)]

    def best(rate):
        """Best (tok/s, k) over draft lengths; rate(k, a) is tokens per ms or None."""
        rates = [(rate(k, a), k) for k, a in accepted.items()]
        rates = [(r * 1000, k) for r, k in rates if r is not None]
        return max(rates) if rates else (None, None)

    rows = []
    for n_users in sorted(map(int, target)):
        base = n_users / T(n_users, 1) * 1000
        coloc = best(lambda k, a: n_users * (a + 1) / (k * draft[str(n_users)]["1"] + T(n_users, k + 1)))
        split = best(lambda k, a: n_users * (a + 1)
                     / (k * split_draft[str(n_users)]["1"] + T(n_users, k + 1) + net))
        half = n_users // 2
        piped = best(lambda k, a: None if n_users < 2 or str(half) not in target else
                     n_users * (a + 1) / max(2 * T(half, k + 1),
                                             T(half, k + 1) + k * split_draft[str(half)]["1"] + net))
        rows.append({"users": n_users, "target_tok_s": base, "colocated": coloc, "split": split,
                     "split_2_groups": piped})

    def cell(result, base, price):
        tok_s, k = result
        if tok_s is None:
            return f"{'-':>28}"
        return f"{tok_s:7.0f} {tok_s / base:5.2f}x k={k} ${price / 3600 / tok_s * 1e6:6.2f}"

    print(f"network {net:.1f} ms/round; accepted per round {accepted}")
    print(f"{'users':>5} {'target alone':>18} {'co-located':>28} {'split':>28} {'split, 2 groups':>28}")
    for r in rows:
        base = r["target_tok_s"]
        print(f"{r['users']:>5} {base:7.0f} ${args.price_target / 3600 / base * 1e6:6.2f}/1M"
              f" {cell(r['colocated'], base, args.price_target)}"
              f" {cell(r['split'], base, args.price_target + args.price_draft)}"
              f" {cell(r['split_2_groups'], base, args.price_target + args.price_draft)}")
    if args.out:
        Path(args.out).write_text(json.dumps({"network_ms_per_round": net, "accepted": accepted,
                                              "rows": rows}, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("measure", help="time decode steps on this GPU")
    p.add_argument("--model", required=True)
    p.add_argument("--batches", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64, 128])
    p.add_argument("--tokens", nargs="+", type=int, default=[1, 2, 4, 6, 9],
                   help="new tokens per step: 1 for drafting and decoding, k+1 for verifying")
    p.add_argument("--context", type=int, default=128, help="cached tokens per sequence")
    p.add_argument("--replays", type=int, default=50)
    p.add_argument("--out", required=True)
    p.set_defaults(func=measure)
    p = sub.add_parser("predict", help="estimate throughput per setup from measured steps")
    p.add_argument("--target", required=True, help="target model on the target GPU")
    p.add_argument("--draft", required=True, help="draft model on the target GPU (co-located)")
    p.add_argument("--split-draft", required=True, help="draft model on its own GPU (split)")
    p.add_argument("--price-target", type=float, default=1.79, help="target GPU $/hour")
    p.add_argument("--price-draft", type=float, default=0.89, help="draft GPU $/hour")
    p.add_argument("--out")
    p.set_defaults(func=predict)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
