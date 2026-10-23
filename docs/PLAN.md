# specwire — Plan

Disaggregated speculative decoding: run the small draft model and the large
target model on separate machines, connected over the network, and measure
honestly when that beats the standard setups.

Reimplementation and continuation of the group project
[SamvritS/SpecSplit](https://github.com/SamvritS/SpecSplit) (MIT). Original
authors get prominent credit in the README.

## Why redo it

The original built most of the system but never answered its own question:

- Benchmarks ran through ngrok tunnels → ~1–2 tokens/sec, ~99.9% of time
  waiting on the network.
- The core claim (network RTT ≪ target verify time) was never validated.
- No target-only baseline, so no speedup number exists.

## Goal

Show with real numbers whether draft/target disaggregation is faster or
cheaper than (a) the target model alone and (b) co-located speculative
decoding, and find the network delay at which it stops paying off.

## Phases

### Phase 0 — Understand the original
- [x] Read `pipeline.py`, `target/engine.py`, `verification.py`, `kv_cache.py`
- [x] Write a one-page summary in my own words

### Phase 1 — Core rebuild (Mac, tiny models)
- [x] Draft worker, target worker, orchestrator over gRPC
- [x] Linear (chain) drafts — no trees yet
- [x] Greedy verification + rejection sampling
- [x] KV cache with rollback on both workers
- [x] Test: greedy spec-decode output == target-only output, token for token

### Phase 2 — Baselines
- [x] Target model alone (plain autoregressive)
- [x] Both models on one GPU (standard speculative decoding)
- [x] Split across machines (the thesis)

### Phase 3 — Cloud benchmarks
Setup: draft (Llama-3.2-1B) on L4, target (Llama-3.1-8B) on A100, same
region / datacenter. Stretch: 70B (4-bit) on A100 80GB.

- [x] Tokens/sec and time-per-token: split vs co-located vs target-only
- [x] Speedup vs draft length (1, 3, 5, 8)
- [x] Speedup vs network delay: same zone, +1/5/10/20 ms injected, cross-region
- [x] Cost per 1M tokens from real hourly prices
- [x] Target-GPU memory freed by moving the draft off it

### Phase 4 — Improvements (pick 1–2)
- [ ] Tree drafting
- [ ] Adaptive draft length
- [ ] Overlap: draft next round while target verifies
- [ ] Multi-user batching (where freed target memory should pay off)

### Phase 5 — Write-up
- [x] README: problem, design diagram, 3–4 charts, honest conclusion
- [x] Credit to SpecSplit and its authors
- [ ] Discuss EAGLE-style drafting and what it means for splitting
- [ ] Short blog post or 2-page report

## Risks / unknowns

- Cost: rough guess $20–60 total (A100 ~$1.5–2/hr, L4 ~$0.5/hr). Unverified.
- Placement: provider must put both machines close together (RunPod / Lambda
  allow choosing a datacenter; Modal may not).
- Result: for 8B, verify time may be too short for splitting to win; 70B is
  more likely to. A clear break-even answer is still a good result.

## Decisions

| Decision          | Choice                                    |
| ----------------- | ----------------------------------------- |
| Name              | specwire                                  |
| Repo              | private on GitHub, public after Phase 1   |
| Compute           | Mac for dev, cloud GPUs for benchmarks    |
| Cloud provider    | RunPod                                    |
