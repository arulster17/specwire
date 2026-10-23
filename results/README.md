# Results

Every benchmark run, as written by `specwire.bench` (or `scripts/batchbench.py` for
the `batch-*` files). Each bench file holds the exact arguments (`args`) and, per
setting, the summary plus every run's raw timings and output tokens. All runs are bf16,
greedy, 4 prompts × 128 tokens, 3 timed runs after 1 warmup, unless noted.

## Local: RTX 5070 Ti, both models on one GPU

| File | Models | What it measures |
|---|---|---|
| `bench-3b-eager.json` | Qwen2.5 0.5B → 3B | target-only, co-located and split at k = 1/3/5/8, before fast mode |
| `bench-3b-fast.json` | Qwen2.5 0.5B → 3B | the same sweep with fast mode (static cache + CUDA graphs) |
| `bench-3b-fast-rtt.json` | Qwen2.5 0.5B → 3B | split at k=3 with 0–20 ms of delay injected per call |
| `bench-llama3b-fast.json` | Llama 3.2 1B → 3B | the fast-mode sweep with Llama models |
| `fp32-check.json` | Qwen2.5 0.5B → 3B | fp32 run confirming output equals target-only on every prompt |

## Cloud: RunPod EUR-IS-1, Llama 3.2 1B → Llama 3.1 8B, fast mode

| File | Draft GPU | What it measures |
|---|---|---|
| `p3-a100-baselines.json` | (A100) | target-only and co-located on the A100, k = 1/3/5/8, with $/1M tokens |
| `p3-split-4090.json` | RTX 4090 | split, same datacenter, k = 1/3/5/8 |
| `p3-split-4090-delay.json` | RTX 4090 | split at k=3 with 0–20 ms injected per call |
| `p3-split-l4.json` | L4 | split, same datacenter, k = 1/3/5/8 (k=5 hit a network spike) |
| `p3-split-l4-k5-rerun.json` | L4 | clean rerun of k=5 |
| `p3-split-l4-delay.json` | L4 | split at k=3 with 0–20 ms injected per call |
| `p3-split-4090-EUR-IS-2.json` | RTX 4090 in EUR-IS-2 | split from another datacenter (68 ms network per round) |
| `p3-split-4090-US-IL-1.json` | RTX 4090 in US-IL-1 | split across the Atlantic (143 ms network per round) |

In `p3-split-4090-EUR-IS-2.json`, the 4090 itself ran slowly during k=1 and k=3
(29–62 ms per draft token instead of about 4.6), so only k=5 and k=8 reflect the network.

## Batch-size microbenchmark (A100 and RTX 4090)

| File | What it holds |
|---|---|
| `batch-a100-8b.json` | 8B decode-step time on the A100, batch 1–128 × 1–9 new tokens |
| `batch-a100-1b.json` | 1B draft step on the A100, batch 1–128 |
| `batch-4090-1b.json` | 1B draft step on the RTX 4090, batch 1–128 |
| `batch-predict.json` | predicted throughput per setup and batch size, from the three above |

`archive/` holds superseded runs: the first local benchmarks and the cloud smoke tests.
