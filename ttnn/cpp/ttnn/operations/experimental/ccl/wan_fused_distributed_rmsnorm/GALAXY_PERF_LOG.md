# wan_fused_distributed_rmsnorm — Wormhole Galaxy read-overlap optimization log

Machine: WH Galaxy (4x8), TP=4 LINE on a 1x4 submesh, 4 fabric links.
Bench: `test_wan_rmsnorm_bench_composite_tp4_ring_galaxy` (fused, traced, 100 iters).
Correctness: `test_wan_rmsnorm_correctness_tp4_galaxy` (fused vs composite PCC, must stay ~0.999).

All times are fused µs/iter (lower = better). Composite reference (committed baseline):
N18944-rope 1244, N9472-rope 609, N2368-rope 198, N18944 1037, N9472 517, N2368 151, L512 75.

## Baseline (commit 5e1eb846dea, fix landed) — fused µs/iter

| config | seq_len | RoPE | fused µs |
|---|---:|:--:|---:|
| self_sp4_N18944    | 18944 | Y | 981.7 |
| self_sp8_N9472     | 9472  | Y | 497.4 |
| self_sp32_N2368    | 2368  | Y | 244.1 |
| cross_q_sp4_N18944 | 18944 | N | 592.7 |
| cross_q_sp8_N9472  | 9472  | N | 321.8 |
| cross_q_sp32_N2368 | 2368  | N | 142.9 |
| cross_k_prompt_L512| 512   | N |  64.3 |

## Ablation findings (exposed cost = baseline − ablation, % of baseline)

| config | rope-rd | input-rd | out-wr | fabric | gather/scatter |
|---|---:|---:|---:|---:|---:|
| N18944 RoPE | 242 (25%) | 280 (29%) | 76 (8%) | 9 (1%) | 4 (0%) |
| N9472 RoPE  | 59 (12%)  | 109 (22%) | 4 (1%)  | 5 (1%) | 9 (2%) |
| N2368 RoPE  | 25 (10%)  | 40 (17%)  | 1 (1%)  | 5 (2%) | 8 (3%) |
| N18944 no-rope | — | 114 (19%) | 46 (8%) | 16 (3%) | 18 (3%) |
| N9472 no-rope  | — | 35 (11%)  | 23 (7%) | ~0 | ~0 |
| N2368 no-rope  | — | 17 (12%)  | 2 (1%)  | 3 (2%) | 7 (5%) |
| L512 no-rope   | — | 2 (3%)    | 3 (5%)  | 1 (1%) | 2 (3%) |

Conclusion: latency-bound reads on the critical path. cos/sin reads (RoPE) and input
reads are the two big exposed costs; fabric + gather/scatter are hidden. Ideas A–F target
read overlap. Each idea below: implement → correctness (PCC) → measure → keep if faster.

---
