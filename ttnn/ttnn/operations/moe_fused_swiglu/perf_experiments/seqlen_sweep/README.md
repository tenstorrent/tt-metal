# Sequence-length sweep — moe_fused_swiglu

Device kernel duration at every tile-aligned token count from 32 to 5120, both supported input
formats. **Blackhole p150**, 110 Tensix @ 1.35 GHz, emb 7168, hidden 2048, capacity 5120,
`bfloat4_b` weights, LoFi. 320 points, median of 3 reps, rep spread ≤ 2.9 % everywhere.

| file | what |
|---|---|
| `seqlen_scaling.png` | the four-panel scaling plot (110 cores, interleaved weights) |
| `seqlen_sweep.csv` / `.json` | one row per (format, count): ns median/min/max, µs, DRAM util, ns/token, tokens/s |
| `wplace_88.png` | weight-placement A/B at 88 cores |
| `sweep88.csv` / `.json` | 9 counts × 2 formats × {interleaved, ND-sharded}, 88 cores, median of 7 |

**Both sweeps use INTERLEAVED weights unless the row says `nd_shard`.** Placement is the caller's
choice, not a knob: `nd_shard_n_tiles()` reads the shard width off the tensors it is handed, so
`DRAM_MEMORY_CONFIG` weights silently take the uncoalesced one-request-per-tile path. The graded
`PERF_MEASURED_NS` baselines come from the same `_build`, so they are interleaved too.

## Reproduce

```bash
# ~8 min per format; each chunk is its own profiled session (see "chunking" below)
perf_experiments/sweep_seqlen.sh bf16_rm   48 7168 5120 32 3
perf_experiments/sweep_seqlen.sh bfp8_tile 48 7168 5120 32 3
perf_experiments/parse_seqlen_sweep.py <out_prefix> $(cat /tmp/moe_seqlen/pairs_*.txt | tr '\n' ' ')
perf_experiments/plot_seqlen_sweep.py <out_prefix>.json <out>.png
```

The swept axis is `count`, the number of real tokens routed to the local expert — this op's sequence
length. `capacity` (allocated slots) is a separate, coarse axis: `SUPPORTED[capacity]` is
`{1024, 2048, 5120}`, so it cannot be swept at 32-token resolution — and it is free, which is why
sweeping `count` at one capacity is the whole curve. Measured at count 256, capacity 1024 / 2048 /
5120 (`measure.sh`, i.e. an independent harness): 143.4 / 143.4 / 141.7 µs `bf16_rm` and
137.7 / 138.9 / 139.2 µs `bfp8_tile` — a ≤ 1.2 % span with no ordering, so allocation costs nothing.
Those also agree with this sweep's own count-256 points (142.4 / 137.5 µs) to within 1 %.

`count` is DEVICE-resident, so **every point runs the same compiled program** — the only thing that
changes between points is the content of the 256-entry `counts` tensor.

## Two traps this harness had to solve

1. **Profiler DRAM buffer.** The op emits up to ~125 zone records per core per dispatch against a
   12000-record device buffer, so a long session drops markers and tracy then aborts report
   generation entirely (`Device data missing: Op N not present in cpp_device_perf_report.csv`) —
   there is no partial CSV. The sweep calls `ttnn.ReadDeviceProfiler` once per point to drain it.
2. **Chunking is a HOST memory limit, not a device one.** tracy's `process_ops_logs` holds the whole
   trace in pandas and needs ~50 MB RSS per profiled dispatch: 964 dispatches in one session reached
   50 GB and were OOM-killed *after every dispatch had already succeeded on device*. Hence
   ~150 dispatches per session, merged in the parser.

## Results

| | bf16 · ROW_MAJOR | bfp8_b · TILE |
|---|---|---|
| fixed floor (tail fit intercept) | 64.3 µs | 63.0 µs |
| marginal cost | 363 ns/token | 340 ns/token |
| count 32 | 81.1 µs | 80.5 µs |
| count 512 | 236.5 µs | 226.0 µs |
| count 5120 | 1913.5 µs | 1795.3 µs |
| peak matmul throughput | 236 TFLOP/s | 251 TFLOP/s |
| peak DRAM read utilisation | 61 % (count 32) | 61 % (count 32) |

## 88 cores (`MOE_SWIGLU_GRID=11x8`) and weight placement

`sweep88.csv`, median of 7. The placement is asserted per run against `nd_shard_n_tiles()` — the
reader's own predicate — so a config that failed to shard fails the test instead of quietly
reporting the interleaved number. Verified widths: `[6, 6, 3]` = `HN_PAD` ⌈64/11⌉ for gate/up,
`EC_MAX` ⌈224/88⌉ for down.

| M | intlv 88c | ND-shard 88c | shard win | intlv 110c | 88 vs 110 |
|---|---|---|---|---|---|
| 32 | 82.27 µs | 74.86 µs | −9.0 %* | 81.08 µs | +1.5 % |
| 64 | 86.25 | 79.93 | −7.3 %* | 85.55 | +0.8 % |
| 128 | 102.51 | 91.34 | −10.9 %* | 100.30 | +2.2 % |
| 256 | 135.37 | 130.67 | −3.5 % | 142.41 | −4.9 % |
| 384 | 189.91 | 182.93 | −3.7 % | 194.00 | −2.1 % |
| 512 | 241.49 | 229.58 | −4.9 %* | 236.49 | +2.1 % |
| 1024 | 429.76 | 425.50 | −1.0 % | 424.37 | +1.3 % |
| 2048 | 821.07 | 816.27 | −0.6 % | 797.53 | +3.0 % |
| 5120 | 2000.08 | 1987.38 | −0.6 %* | 1913.49 | +4.5 % |

`bf16_rm`; `bfp8_tile` is in the CSV and behaves the same (−9.9 % at M=32, −0.4 % at M=5120).
`*` = the delta exceeds the interleaved baseline's own full rep spread. Below 512 tokens that
baseline spreads 3–6 % run-to-run while the sharded one spreads < 1 %, so the mid-range deltas
(M=256, 384) are not separable from noise and only the ~10 % short-sequence win and the ~0 %
long-sequence result are load-bearing.

**Dropping 110 → 88 cores costs 0.8–4.5 %, not 20 %.** The op is not compute-throughput-limited at
110 cores. Two cells (M=256, 384) are *faster* at 88 cores; both sit in the noisy band above, so read
them as "no worse", not as a real inversion.

## Why the large-M asymptote is what it is

The tail is linear with a small intercept, so the *shape* is right; what is poor is the constant —
363 ns/token (110 c, bf16), i.e. 242 TFLOP/s, and it stops improving past ~2k tokens. Three
measurements say the steady state is not math and not bandwidth:

| probe | marginal cost, M 1024 → 5120 | reading |
|---|---|---|
| baseline | 363.0 ns/token | (sweep says 363.6 — 0.2 % agreement) |
| `MOE_SWIGLU_ABLATE=skip_compute` | 261.2 ns/token | removing ALL matmul math buys only **28 %** |
| `MOE_SWIGLU_ABLATE=no_h_xfer` | 295.7 ns/token | the h all-gather *payload* is 18.5 % |

Plus: DRAM traffic at M=5120 is 137 MB in 1913 µs = **14 % of 512 GB/s**, and **78 % of the marginal
token cost is insensitive to core count** (slope 363 ns at 110 c vs 383 ns at 88 c — 1.055×, where
compute-bound would be 1.250×).

So ~72 % of the marginal per-token cost is dataflow + rendezvous. That matches the mechanism Perf 15
measured at count 256: phase 2 is 83 % one `noc_semaphore_wait` on the h arrival flag, waiting on
phase 1's tail, and with every payload ablated phase 2 is still ~43 µs ≈ 11 rounds × ~3.9 µs of pure
grid-wide serialisation — against a 93 µs marginal cost per 256-token M-block. **Rounds per token is
constant, so that cost never amortises**: more tokens buy more rounds, which is precisely why the
per-token curve floors instead of continuing to fall.

The lever would be fewer rendezvous per token — fatter M-blocks or deeper h pipelining — and both are
L1-blocked: `M_BLOCK=16` needs ~400 KB more than the 121 728 B free at 11×10 (10 560 B at 11×8 —
always state the grid with an L1 figure for this op), and `DEPTH_H=4` is a measured null.

**Latency is a step function of work, not of tokens.** Cost is linear in
`work_rows(count) = 8·⌊M_t/8⌋ + next_pow2(M_t mod 8)` where `M_t = count/32` — the descriptor's
`M_BLOCK = 8` plus its power-of-two tail rounding (`m_tiles_eff`). Over the 159 consecutive 32-token
steps of the sweep, **80 are free** (|Δ| ≤ 3.3 µs, and every one of them has zero work-row change)
and the rest cost ~13 / ~19 / ~40 µs for a 1 / 2 / 4 work-row increase: 159/159 steps consistent for
`bfp8_tile`, 158/159 for `bf16_rm`. Consequence for a caller who can choose its padding: rounding
`count` up to a work-row plateau is free, e.g. 4768 → 4864 tokens costs +0 µs, while a single
32-token step across a boundary (4224 → 4256) costs +42 µs.
