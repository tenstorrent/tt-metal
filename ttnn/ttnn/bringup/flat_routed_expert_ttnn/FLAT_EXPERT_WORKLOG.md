# Flat streamed routed expert: generalization / raggedness work log

Op under test: `test_stream_expert_flat.py` (+ `kernels/stream_mm/se*`), one Blackhole p150a, bfp4 weights, LoFi.
"Hard case" = row-major bf16 dispatch buffer in DRAM (`MIMO_FL_DYN=1`, implies E2E), per-expert token counts read
on device, output bfp8 tiles at each expert's region.

Timing: Tracy device-kernel duration between signposts, mean of 3 iterations (`analyze_tags.py`). "us" below is
the whole launch (all local experts), not per expert, unless marked.

## 2026-09-27

### Setup
- Workload reference: Kimi K2.6 routing dump (artifact ad3ca641, 60 layers x 11 chunks x 384 experts, 5120 tok/chunk,
  top-8, 32 chips x 12 local experts). Per chip: mean 1280 tokens, p50 1153, p90 1910, p99 3442, max 6139; per
  expert: mean 107, p50 69, p99 756, max 3775; within a chip max/mean expert 3.1x (p50), 5.7x (p90), 8.8x (p99).
  Program capacity must cover the max -> built with M (capacity) = 4096, E = 12.
- Harness changes (DYN mode): count sets (`MIMO_FL_COUNTS "a,b;c,d"` or `MIMO_FL_COUNTS_FILE` json
  [[label, counts]]) run back to back on one program (same buffers, contents rewritten); regions packed like a real
  dispatch buffer (32-row aligned prefix sums) instead of `e * capacity`; pre-tiled x no longer built in E2E mode;
  PCC checked on up to 160 rows at each end of every expert (`MIMO_FL_CHECK_ROWS`).
- Shapes survey (repo): K2.6/K2.7/DSv3 7168x2048 SiLU; Kimi K3 3584x3072 SiTU-GLU; MiniMax-M3 6144x3072 swigluoai;
  MiniMax-M2.7 3072x1536; GLM-5.x 6144x2048; DSv4-Flash 4096x2048 / Pro 7168x3072 clamped_silu_glu;
  Mistral-S4 4096x2048; Qwen3.5-397B 4096x1024; GPT-OSS 2880x2880 swigluoai + biases; Gemma-4 26B 2816x704 GeGLU.
  Activation reference math: unified_routed_expert_ffn/.../compute/fused_swiglu.cpp.

### Smoke (K2 7168x2048, E=12, capacity 4096)
| counts | tokens | us |
|---|---|---|
| balanced 12 x ~107 | 1280 | 825.2 |
| real p50 chip [74,196,148,67,65,94,330,65,34,41,10,29] | 1153 | 836.5 |
PCC min 0.9943 (quantized-weight reference).

### Baseline ragged sweep (K2, E=12, capacity 4096, before any change)
Real chips from the K2.6 dump (tot-pN = chip at the N-th percentile of total tokens, max-pN = of its largest expert),
each vs a "balanced twin" (same total spread evenly over the 12 experts), plus uniform / synthetic sets.

| set | counts | tokens | ragged us | balanced us | ragged / bal |
|---|---|---|---|---|---|
| tot-p50 | 74,196,148,67,65,94,330,65,34,41,10,29 | 1153 | 836.9 | 814.8 | +2.7% |
| tot-p90 | 117,108,241,66,19,1000,43,40,87,48,34,107 | 1910 | 957.2 | 930.0 | +2.9% |
| tot-p99 | 2876,49,90,49,72,21,24,73,68,55,41,26 | 3444 | 1277.9 | 1146.9 | +11.4% |
| tot-p100 | 13,1050,30,30,36,117,66,29,725,3674,350,19 | 6139 | 1714.2 | 1469.4 | +16.7% |
| max-p50 | 93,46,19,39,289,91,43,84,4,50,94,94 | 946 | 810.5 | 801.0 | +1.2% |
| max-p99 | 314,23,31,157,8,85,151,42,17,2320,33,77 | 3258 | 1245.5 | 1150.4 | +8.3% |
| max-p100 | 12,1189,11,47,25,65,73,50,468,3775,104,6 | 5825 | 1674.1 | 1460.1 | +14.7% |

Uniform (12 x M): M 32/64/128/192/256/384/512/1024 = 770/796/839/945/1001/1199/1463/2482 us -> per expert 64 us at
the weight floor (24.8 MB bf4 at ~390 GB/s), incremental 0.166 us/token above M=512 (~530 TFLOP/s, 87% of LoFi peak).
Synthetic: spike1024 [1024 + 11 x 24] 910; spike3000 [3000 + 11 x 25] 1285; two-hot [1500,1500 + 10 x 30] 1232.

Diagnosis: a big expert is compute-bound (DRAM mostly idle) while every small expert costs the full weight floor
(~64 us) with compute idle; the gate/up ring holds 2 experts so only ONE expert's weights prefetch under the big
expert. Sum-of-parts model (per-expert cost from the uniform curve) reproduces the ragged numbers within ~5%.

### Pinned schedule (se_dyn.hpp `SE_PIN_MIN`, host `MIMO_FL_PIN`)
- Schedule = list of entries (runs of sub-blocks of one expert); the biggest expert B is cut into K chunks
  interleaved with the small experts (<= `SE_PIN_SMALL` = 2 sub-blocks): B1 s1 B2 s2 .. BK sK .., then the larger
  experts. B's weights load once and stay in ring region 0 until BK; the small experts cycle through region 1, so
  each small expert's weights stream in while a chunk of B computes. Everything but the weight path (x relays, h,
  y, down compute counts) just sees more entries.
- Ring mechanics: loads land in regions; receivers keep granting slots by count; compute pops by count after a
  block's last use and addresses block (region, j) relative to the read pointer (popped % ring). Load l goes to the
  region of the (l - NREG)-th load to retire (retirement order = schedule order of last uses); the builder checks
  that retirement comes before the load's first use, else drops pinning. (That check models retirement order, not
  producer batching: see the codex review below, batched down reads could straddle a load boundary and deadlock.)
- Down weights too (`SE_DN_REG`): pinning only the gate/up weights re-read B's down weights per chunk (8.3 MB) and
  gained just 5%; pinned down rings need 2 whole experts (DRING 2.0, default with PIN) -> did not fit L1 (1434 KB
  vs 1427 KB/bank) until the e2e output double buffer was sized for bfp8 pages (was 2 KB pages: -53 KB, arena 1382 KB).
- Row tiles: a sub-block's compute (gate/up rows, down row passes) now skips row tiles past the token count (the
  last sub-block of each entry: `SE_META_LMT`); small experts cost 1-3 row tiles instead of 4.
- Profiling (p99 chip): gate/up MM and down MM both ~24 us per full 128-row sub-block (~18 cyc/tile = LoFi peak);
  small-expert down 6 us per row tile. After pinning the p99 chip is within ~20% of an IDEAL DRAM floor (12 x 24.8
  MB weights + 75 MB logical bf16 x / bf8 y at ~420 GB/s ~ 890 us) vs 1068 measured; this implementation actually
  moves more (the two rectangle relay teams each read x: +49 MB), so its own floor is ~1010 us.
- Profiler gotcha: `SE_DOWN` in se3_compute.cpp (unused down path) hash-collided with BRISC-FW once line numbers
  moved -> renamed `SE_DOWN3`.
- Tried, no gain: `MIMO_FL_DW_DELAY` 40k/80k cycles (down weights start late so expert 0's gate/up weights own DRAM
  during fill): within +-1%.

Results (PIN=1, SMALL=2; same program and capacity for every set):
| set | before | pinned | balanced twin (pinned build) | ragged / bal |
|---|---|---|---|---|
| tot-p50 | 836.9 | 821.3 | 814.8 | +0.8% |
| tot-p90 | 957.2 | 904.3 | 943.6 | -4.2% |
| tot-p99 | 1277.9 | 1068.1 | 1176.3 | -9.2% |
| tot-p100 | 1714.2 | 1590.6 | 1456.7 | +9.2% |
| max-p50 | 810.5 | 809.6 | 806.4 | +0.4% |
| max-p99 | 1245.5 | 1101.8 | 1165.7 | -5.5% |
| max-p100 | 1674.1 | 1543.1 | 1464.3 | +5.4% |
| spike1024 | 909.8 | 834.7 | | |
| spike3000 | 1285.0 | 1106.9 | | |
| two-hot | 1232.3 | 1169.5 | | |
Uniform 12 x M unchanged within noise (32: 754, 128: 845, 512: 1467, 1024: 2466). Interleaving every other expert
(not only small ones) was 1562 on tot-p100 but cost 2-3% on uniform 384/1024 (chunking between equally big experts
only adds switches) -> small-only interleave. PIN 1 vs 2 vs 3: p99 1122 / 1128 / 1136 (before row-tile skipping).
Balanced 12 x 512 profile: 23.6 us per sub-block steady; ~140 us launch fill (expert 0's gate/up weights share DRAM
with 16.6 MB of down weights), boundary jitter from DRAM at ~85-90% (weights + bf16 x + y ~375 GB/s).

### Activation functions (se3_compute.cpp `SE_ACT`, host `MIMO_FL_ACT`)
Run on the PACK thread's SFPU over the raw gate / up DST accumulators (as SiLU was, so MATH starts the next
sub-block's matmul): the unified_routed_expert_ffn variants, same SFPU functions, invoked from PACK:
`silu` (0), `swigluoai` (1, GPT-OSS / MiniMax-M3: (clamp(u,+-7)+1) g' sigmoid(1.702 g'), g' = min(g, 7), via
moe_gpt's swiglu_sfpu.h), `situ` (2, Kimi K3: 4 tanh(g/4) sigmoid(g) * 25 tanh(u/25), calculate_situ_glu),
`clamped_silu` (3, DeepSeek V4: silu(min(g,10)) clamp(u,+-10)), `gelu_tanh` (4, Gemma 4: gelu_tanh_tile_pack + mul).
Check is discriminating: the test also computes the PCC against the SiLU-GLU reference and asserts the chosen
activation's reference matches strictly better (weights scaled with `MIMO_FL_WSTD` so the clamps / tanh engage:
0.06 -> gate std ~5; GeGLU vs SiLU-GLU only separate at small gates: 0.006).
| act | min PCC (own ref) | min PCC (silu-glu ref) | uni128 us | tot-p99 us |
|---|---|---|---|---|
| silu | 0.9947 | - | 844.1 | 1093.9 |
| swigluoai | 0.9923 | 0.9021 | 844.0 | 1084.3 |
| situ | 0.9935 | 0.9328 | 848.5 | 1077.9 |
| clamped_silu | 0.9928 | 0.9751 | 843.7 | 1089.2 |
| gelu_tanh | 0.9938 | 0.9777 | 848.0 | 1080.5 |
No measurable cost on these two workloads at this precision (single runs, the activation hides under the next
matmul). Not done: GPT-OSS gate/up/down biases.

### Shape generalization
- `MIMO_FL_I` (per-device intermediate size, I / TP) and any H with Ht % 8 == 0.
- Gate/up split `_gu_split`: NP column pairs per gate/up core x G M-groups (group g computes row tiles
  g*MT/G.. of every sub-block; the x block still carries all MT rows, compute starts at row offset g*MT/G). The
  It/NP pair-sets go round the 16 readers (multiple of 16), each forwarded to its G cores (forwarder receiver j gets
  block j / G). Picks the most cores (<= 64), then the fewest pairs per core, subject to DST ((MT/G) * 2 NP <= 8)
  and gate/up L1 (2-expert ring of the pairs' full-K slice + x ring <= 1400 KB; the x ring shrinks 24 -> 16 -> 12
  slots first). Unused gate/up cores idle inside the multicast rectangles (arena allocated there).
  I 2048: NP1 G1 (64 cores); 1024: NP1 G2; 512: NP1 G4; 3072 @ H 6144/7168: NP2 G1 (48 cores, MT 2);
  3072 @ H 3584: NP3 G2 (64, MT 2); 1536: NP1 G1 (48 cores).
- Relays: super-block width `SE_SBT` (K tiles per row-major chunk: 32, 16 for H 3584).
- Down: HBUF drops to 2 when 3 h buffers + the (pinned, 2-expert) down ring overflow L1 (I 3072).
- Bugs found on the way: (1) se5_recv read RT 14 both as the M-group and as the old pair role (`is_b`), so group-1
  cores never granted weight credits -> hang (DPRINT of receiver/forwarder counters: credits 56 0 56 0);
  (2) the forwarder sent a whole block as one NoC packet; NP 2 blocks are 18 KB > NOC_MAX_BURST_SIZE (16 KB) -> hang;
  now bursts under one trid; (3) my own L1 check at 1300 KB silently moved K2 from 24 to 16 x slots (fixed: 1400).
- Also fixed: an indentation slip in the e2e check block that had skipped the per-expert PCC assert for SiLU runs
  (introduced with the activation check; only two H-sweep runs affected, rerun).

### TP shapes: M-groups vs fewer cores; x path
- G > 1 re-forwards every gate/up weight block to G cores. G = 1 on fewer cores (K2 TP2: 32, TP4: 16) is faster at
  small M and equal at large M (x-path bound anyway), 8 experts, us:
  | shape | G>1 uni32 / 128 / 512 / rag | G=1 uni32 / 128 / 512 / rag |
  |---|---|---|
  | K2 TP2 7168x1024 | 353 / 398 / 826 / 612 (G2) | 283 / 341 / 826 / 603 |
  | K2 TP4 7168x512 | 306 / 343 / 807 / 584 (G4) | 183 / 237 / 794 / 560 |
  | K3 3584x3072 | 453 / 540 / 1130 / 887 (NP3 G2) | 403 / 464 / 988 / 752 (NP2, 48 cores) |
  -> `_gu_split` now prefers the fewest M-groups.
- Large-M TP profile (TP4, 8 x 512, 794 us): gate/up and down compute ~6 us per 128-row sub-block but the pipeline
  runs at ~25 us; the relays' tilizer is the limit: `fast_tilize_block` of 32 tiles to bfp8 ~2400-2800 cycles
  (~80 cyc/tile), each relay (primary + helper per rectangle) tilizes 448 tiles per sub-block, and every x tile is
  tilized twice (once per rectangle).
- Tried: shared tilizing (XSHARE: the 4 relays each tilize every 4th super-block once and write it to BOTH
  primaries' landing rings): correct (PCC ok) but 1.6-1.8x SLOWER (8 x 512: 1261 vs 794 us; 8 x 32: 333 vs 183).
  Zones: primaries wait 450-660 us per launch for landed super-blocks; the helpers' 139 KB unicasts now cross the
  chip (6 hops, NOC0, through the rows the linked multicasts reserve) instead of going to the adjacent primary. A
  deeper landing ring (6 slots, 2 own) changed nothing. Reverted.
- Re-diagnosis (the relay is not the whole story): with G = 1 each of the 16 (TP4) gate/up cores does a full
  128-row sub-block for its pair, gate AND up: 4 x 224 x 2 = 1792 tile-matmuls ~24 us -> compute-bound on 16 cores;
  with G = 4 the 4x duplicated forwarding (16.5 MB per expert over its 4 sub-blocks at M = 512) is also ~25 us.
  Probes that changed nothing at 8 x 512 (794 us): DRAM x reads off (`MIMO_FL_XRD_SKIP=1`: 791), contiguous 35 KB
  multicast pieces instead of 8.7 KB (795), h chain pieces 32 -> 8 / 4 (793 / 794), 2 helpers per rectangle (797).
- Tried: multicast forwarding for G > 1 (each block once to its G cores' 1x2 / 2x2 / 1x4 rectangle). With a
  transaction id on the multicast -> HANG (the per-id outstanding counters do not track multicast acks: confirms the
  earlier trid gotcha). Without trid (chunks complete in batches when all this RISC's writes are acked, 2-8 deep):
  correct but 2.1x slower at small M (8 x 32: 389-414 vs 183 us), ~4% faster at 512 (754 vs 794): 16 forwarders'
  path-reserved multicasts on NOC1 contend. Reverted.
- New: `MIMO_FL_XHELP_N` helpers per rectangle (se11_xmc round of 1 + NH, per-helper landing ring + arrival sem;
  2 needs `MIMO_FL_LAND_SLOTS=2` to fit L1). Best large-M TP4 point: G 2 (32 cores, 2x forwarding) + 2 helpers:
  | K2 TP4, 8 experts | uni32 | uni128 | uni512 | rag |
  |---|---|---|---|---|
  | G1 (default) | 183 | 236 | 794 | 560 |
  | G2 | 199 | 247 | 732 | 519 |
  | G2 + 2 helpers | 198 | 253 | 675 | 520 |
  | G4 + 2 helpers | 285 | 344 | 756 | 569 |
  Real routing is small-expert dominated, so G 1 stays the default; G 2 + 2 helpers is the opt-in for large M.

### Regression found by the E / token study: auto-HBUF margin
The study (below) showed PIN=1 13-19% slower than PIN=0 at 512 tokens/expert even for balanced counts (no pinning
happens there): my auto-HBUF rule kept a 128 KB margin, so with PIN's 2-expert down ring the down cores silently
dropped to HBUF 2 (the K2 arena with HBUF 3 is 1382 KB of 1427 and always fit). Margin removed; E8 bal512
1176 -> 1029 us (PIN0 1018). All PIN=1 numbers taken since the auto-HBUF commit (e417c68: the 12-shape matrix, the
TP G=1/G=2 tables) carry it at large M; rerun below. Small-M numbers (weight-bound) were unaffected.

### Tokens x experts x raggedness study (K2 7168x2048, bf4, capacity 4096, pinned build, after the HBUF fix)
Sets per E: mean 32 / 128 / 512 tokens per expert; balanced, zipf (1/i), spike (one expert half the tokens, rest
equal), real (E counts sampled from the pooled K2.6 per-expert distribution, rescaled). us per launch; ratio to
the balanced set of the same total.
| E | mean | bal | zipf | spike | real |
|---|---|---|---|---|---|
| 4 | 32 | 272 | 272 (1.00) | 274 (1.01) | 280 (1.03) |
| 4 | 128 | 316 | 301 (0.95) | 313 (0.99) | 320 (1.01) |
| 4 | 512 | 570 | 572 (1.00) | 585 (1.03) | 580 (1.02) |
| 8 | 32 | 511 | 516 (1.01) | 520 (1.02) | 529 (1.04) |
| 8 | 128 | 581 | 577 (0.99) | 587 (1.01) | 595 (1.02) |
| 8 | 512 | 1006 | 1022 (1.02) | 1100 (1.09) | 1082 (1.08) |
| 16 | 32 | 996 | 1010 (1.01) | 1013 (1.02) | 1023 (1.03) |
| 16 | 128 | 1110 | 1174 (1.06) | 1146 (1.03) | 1189 (1.07) |
| 16 | 512 | 1914 | 2032 (1.06) | 2181 (1.14) | 2067 (1.08) |
| 28 | 32 | 1719 | 1747 (1.02) | 1754 (1.02) | 1685 (0.98) |
| 28 | 128 | 1910 | 2043 (1.07) | 1967 (1.03) | 2032 (1.06) |
| 28 | 512 | 3249 | 3577 (1.10) | 3360 (1.03) | 3524 (1.08) |
PIN=0 for comparison (same sets): E8 real512 1118, spike512 1105; E16 real512 2071, zipf128 1152; E28 zipf512
3627, real512 3524 -> pinning helps most large-mean ragged sets, but NOT everywhere: E16 zipf128 is 1174 pinned vs
1152 unpinned (+2%); single measurements (mean of 3 iterations, no variance estimate), so +-1-2% differences are
within noise.
- Takeaways: the per-launch cost is set by the weight floor (~64 us/expert) for small experts and by compute
  (~24 us per 128-row sub-block) for large ones; raggedness costs <= 4% at mean 32, -5..+7% at mean 128 and
  +1..+14% at mean 512 (worst: E16 spike512).
- Remaining ragged cost is sub-block granularity, not weight streaming: E16 spike512 = 1 x 4096 + 15 x 273 tokens
  -> 77 sub-blocks vs 64; the partial tail sub-blocks (17 rows = 1 row tile) still cost most of a full sub-block on
  the gate/up side (the K loop streams 2 weight tiles per K step whatever the rows). Interleaving threshold
  `MIMO_FL_PIN_SMALL` 3 / 4 changed nothing (2177-2183 us), profile: the relays wait on gate/up slots.
  Candidate next step: 64-row sub-blocks for tails (MT 2 costs ~5% balanced, H 6144 test) or packing tails of
  several experts into one sub-block (needs per-row expert weights: not possible in one matmul).

### Shape matrix (8 experts, capacity 4096, pinned, rerun after the HBUF fix), us per launch
| model shape (per device) | act | split | uni32 | uni128 | uni512 | rag [2000,40,90,60,30,300,20,50] |
|---|---|---|---|---|---|---|
| K2 / DSv3 7168x2048 | silu | NP1 G1 64 | 511 | 579 | 1024 | 824 |
| K2 TP2 7168x1024 | silu | NP1 G1 32 | 283 | 340 | 827 | 601 |
| K2 TP4 7168x512 | silu | NP1 G1 16 | 182 | 238 | 794 | 560 |
| GLM-5 6144x2048 | silu | NP1 G1 64 | 441 | 502 | 924 | 760 |
| MiniMax-M3 6144x3072 | swigluoai | NP2 G1 48, MT2 | 645 | 712 | 1411 | 996 |
| MiniMax-M3 TP2 6144x1536 | swigluoai | NP1 G1 48 | 341 | 399 | 737 | 581 |
| DSv4-Flash 4096x2048 | clamped_silu | NP1 G1 64 | 287 | 332 | 714 | 574 |
| DSv4-Pro 7168x3072 | clamped_silu | NP2 G1 48, MT2 | 758 | 827 | 1660 | 1189 |
| MiniMax-M2.7 3072x1536 | silu | NP1 G1 48 | 182 | 213 | 537 | 409 |
| Qwen3.5 4096x1024 | silu | NP1 G1 32 | 162 | 197 | 489 | 352 |
| Kimi K3 3584x3072 | situ | NP2 G1 48, MT2 | 403 | 463 | 997 | 744 |
| Kimi K3 TP2 3584x1536 | situ | NP1 G1 48 | 203 | 243 | 619 | 452 |
uni32 per expert vs the bf4 weight floor at 512 GB/s: K2 64 us (24.8 MB: 76% of peak BW), GLM5 55 (21.2 MB, 75%),
M3 81 (31.9 MB, 77%), DSv4P 95 (37.2 MB, 77%), DSv4F 36 (14.2 MB, 77%), K3 50 (18.6 MB, 72%); the TP / small-I
shapes sit lower (K2 TP4 23 us for 6.2 MB: 53%, Qwen3.5 20 us for 7.1 MB: 69%): the per-expert fixed costs
(pipeline fill, h exchange) weigh more when the weights are small. Every shape passes the per-expert PCC >= 0.99
check (0.9942-0.9966) and the discriminating activation check where the activation differs from SiLU-GLU.
Not supported: GPT-OSS 2880 (Ht 90: KBLK 8 does not divide it; also needs gate/up/down biases), Gemma-4 704
(It 22), I/TP pair-set counts that are not a multiple of 16 (e.g. 3072 / 4 = 768).

### Strict review (codex --yolo, 2026-09-27) and fixes
Codex found no counterexample for the ring / pinning mechanism itself (10,000 randomized schedules in a symbolic
simulation, block-wise and whole-load retirement; the `p + 1 > popped` guard and the unsigned relative unpacker
offset are sound). Findings and what was done:
1. P1 counts truncated at 65,536 (`cnt` was made uint16 while `subs` used 32 bits -> schedules disagree, hang).
   Fixed: `cnt` / `subs` uint32 again.
2. P1 static (non-dynamic) M-groups used se_forward.cpp, which has no receiver -> block (j / G) map: wrong weights.
   Fixed: G > 1 always uses the pipelined forwarder (se10_fwd).
3. P1 PREPASS with 16-tile super-blocks hangs (se12_xwr.cpp hardcodes 32). Fixed: PREPASS asserts SBT == 32.
4. P2 pinned down rings + DW_BATCH 3 deadlock (counts 1024,32,32, EARLY_POP 0: 63 blocks published, the next batch
   needs 3 slots, only 1 frees before the pinned expert retires). Fixed: in SE_DN_REG mode a down-weight batch
   never straddles a load boundary (se6_dw.cpp); the case now runs (validation below).
5. P1 validation gap: every expert had the same weights (a wrong expert id / region / stale pinned block would pass).
   Fixed: independent weights per expert (`MIMO_FL_DISTINCT_W`, default 1) and per-expert references.
6. P2 checks: now every row of experts up to 512 tokens and every other row tile (all sub-blocks, all pinned chunks)
   plus the last 32 rows above that; a norm-ratio / relative-error bound (PCC is scale-blind); the last measured
   launch must be bit-identical to the checked one; empty active sets no longer crash the activation check. The
   activation discrimination still only asserts when the two references differ (clamps idle at small weights):
   the activation runs use `MIMO_FL_WSTD` to engage them.
7. P2 stats: tags now carry I / NP,G / PIN / helpers / HBUF; weight bytes count only the active experts; e2e x / y
   bytes use bf16 / bfp8 (were swapped); the effective config is recorded; the 2x relay x reads are noted.
Work-log claims qualified above (pinning "everywhere", raggedness ranges, the p99 floor, activation cost).

**The magnitude check found a real accuracy bug (pre-existing, all shapes): output norm 1.27x the quantized-weight
reference** at PCC 0.994 (rel err 0.30), identical with pinning off, with repeated weights, in the static
pre-tiled path (1.25) and without the activation (1.11, gate only). Cause: bf16 DEST accumulation (ties-away
rounding) over the long gate/up K: plain `ttnn.matmul` LoFi bfp8 x bfp4 (test_dest_gain_probe.py) gives norm
ratio 1.023 at K 2048 and 1.085 at K 7168 with bf16 DEST, 1.000 with fp32 DEST; gate and up each gain ~8.5%,
silu(g)*u compounds it, down adds ~2%. Fix: fp32 DEST accumulation for the gate/up compute (`MIMO_FL_GU_FP32`,
default 1; DST half = 4 tiles -> MT 2 for NP 1): norm ratio 1.27 -> 1.041, rel err 0.30 -> 0.074, perf K2.6 E12:
p50 826 (821), p99 1087 (1068), p100 1599 (1591), uni512 1519 (1467, +3.5%), spike3000 1061 (1107). The
remaining ~4% is the K 2048 down projection (bf16 DEST, pcd up to 8 columns per DST pass) + bfp8 h.
All numbers in the tables above were taken with bf16 DEST gate/up (norm gain ~1.27); timing conclusions stand.

### Post-review validation (distinct weights, all new checks, fp32 gate/up DEST), all PASS
Per-expert PCC 0.9979-0.9984 (was ~0.994 with bf16 DEST), norm ratio 1.02-1.05 (was ~1.27), last launch
bit-identical to the checked one. us per launch; in brackets the bf16-DEST number from the tables above.
| config | results |
|---|---|
| K2 E12 K2.6 sets | p50 828 (821), p99 1106 (1068), p100 1596 (1591), max-p100 1549 (1543), spike3000 1075 (1107), uni512 1515 (1467), p100-bal 1524 (1457) |
| codex deadlock case (1024,32,32; DW_BATCH 3; EARLY_POP 0) | 360, no hang |
| K2 E28 | bal512 3472 (3249), zipf512 3815 (3577), spike512 3736 (3360), real512 3613 (3524), zipf128 1953, real128 1958 |
| K2 TP4 8 experts, G1 | uni32 168 (182), uni128 257 (238), uni512 900 (794), rag 602 (560) |
| K2 TP4 G2 + 2 helpers | 198 / 254 / 669 / 517 |
| K3 3584x3072 situ | 393 (403) / 469 (463) / 1255 (997) / 871 (744) |
| M3 6144x3072 swigluoai | 638 / 705 / 1463 (1411) / 1048 (996) |
| DSv4-Flash clamped_silu | 277 / 334 / 786 (714) / 580 (574) |
| 4096x1024 gelu_tanh | 157 / 200 / 552 / 400 |
| static G2 (se10_fwd forced) | 424, PCC 0.9945 |
fp32 gate/up costs where the 4-tile DST half forces smaller sub-blocks: +3-6% on K2 / M3, +10-13% DSv4-F / TP4 at
512 per expert, +26% on K3 (NP 2 -> MT 1: 32-row sub-blocks). Accuracy first: default on; `MIMO_FL_GU_FP32=0` gives
the old speed with the ~1.27 norm gain. Candidate follow-up: split the gate/up DST into two passes over the K loop
(rows 0-1, 2-3) to keep 128-row sub-blocks with fp32, and fp32 for down (pcd column halves).

### Same bias in the existing routed experts (unified / fused), 2026-09-27
test_expert_ref_k2.py now has `MIMO_KREF_CHECK=1` (per-expert norm ratio / rel err / PCC vs the quantized-weight
reference, weights quantized in the op's W^T layout). K2 7168x2048, bf4, 2 experts x 128 tokens, both paths bf16 DEST
(fused hard-codes kFp32DestAccEn = false; unified runs packer_l1_acc = True, fp32_dest_acc_en = False):
| input / x std | unified norm ratio (PCC) | fused norm ratio (PCC) |
|---|---|---|
| tiled bfp8, 0.1 | 1.016 (0.99958) | 1.021 (0.99941) |
| tiled bfp8, 1.0 | 1.021 (0.99955) | 1.032 (0.99946) |
| row-major bf16, 0.1 | 1.027 (0.99948) | 1.027 (0.99936) |
| row-major bf16, 1.0 | 1.033 (0.99944) | 1.038 (0.99941) |
Flat expert (x std 1.0, row-major): 1.27 (0.994) with bf16 DEST, 1.04 (0.998) with fp32 gate/up DEST.
Why the references are ~10x less biased: they accumulate K in blocks through L1 (packer L1 accumulation), the flat
kernel accumulated all of K in DEST. Plain ttnn.matmul (test_dest_gain_probe.py, K blocks of 8 tiles, 4 x 8 cores),
norm ratio: bf16 DEST without packer L1 acc 1.0230 / 1.0845 at K 2048 / 7168, WITH packer L1 acc 1.0079 / 1.0085,
fp32 DEST 1.0000 either way. So packer L1 accumulation per K-block (8 bf16 DST tiles, 128-row sub-blocks kept) is a
cheaper alternative to fp32 DEST for the flat gate/up (~1.01 per matmul instead of 1.00; the final sum then has to
come back from L1 to DST for the SFPU activation, as the fused op does).

### Packer-L1 gate/up accumulation (`MIMO_FL_GU_ACC=l1acc`, `MIMO_FL_GU_L1ACC_GRP` K-blocks per pass), A/B
se3_compute `SE_GU_L1ACC`: K in passes of GRP K-blocks accumulated in bf16 DST, each pass packed into a bf16
partials CB (arena, c_5) with packer L1 accumulation (first pass overwrites) + a packer drain before the next
pass; the sum is copied back to DST (srcA reconfig to the partials) for the activation. Keeps the 8-tile DST half
(no MT reduction). (First try used a static CB: clashed with the arena on the M3 layout; moved into the arena.)
Same commit, 8 experts, us uni512 / rag, max norm ratio, min PCC (uni32 / uni128 within ~2% across modes):
| shape | bf16 | fp32 DEST | l1acc 1 | l1acc 4 | l1acc 7 |
|---|---|---|---|---|---|
| K3 3584x3072 | 998 / 740, 1.13, .9966 | 1249 / 872, 1.05, .9979 | 1361 / 913, 1.06, .9978 | 1205 / 811, 1.08, .9976 | 1051 / 732, 1.10, .9973 |
| K2 TP4 7168x512 | 794 / 560, 1.26, .9946 | 900 / 602, 1.02, .9983 | 962 / 656, 1.04, .9982 | 885 / 610, 1.07, .9980 | 814 / 576, 1.09, .9977 |
| DSv4-F 4096x2048 | 717 / 570, 1.16, .9965 | 778 / 585, 1.04, .9981 | 716 / 590, 1.05, .9981 | 718 / 595, 1.08, .9978 | 711 / 584, 1.09, .9976 |
| M3 6144x3072 | 1412 / 999, 1.16, .9937 | 1467 / 1060, 1.05, .9979 | 1741 / 1190, 1.06, .9978 | 1587 / 1090, 1.08, .9975 | 1455 / 1028, 1.09, .9971 |
K2 7168x2048 E12 (K2.6 sets): l1acc 1 ~ fp32 (uni512 1524 vs 1521, p100 1641 vs 1612), norm 1.058 vs 1.042.
Conclusion: at equal accuracy (1-K-block passes ~ the probe's 1.009 per matmul) L1 accumulation is SLOWER than fp32
DEST on the NP-2 / per-core-heavy shapes (K3 +9%, M3 +19%, TP4 +7%), faster only on DSv4-F (-8%); longer passes
buy speed back but the bias returns (1.09-1.10 at 7 K-blocks). Each pass packs 8 tiles with an L1 read-modify-write
and a packer drain, comparable to the pass's matmul on these shapes. fp32 DEST stays the default; l1acc is kept as
an option. Untried: ping-pong partial buffers (pass g into buffer g % 2) to drop the per-pass drain, summed once.

### Targeting K3 and TP4 at large M (fp32 gate/up), vs the existing ops
- K3 cost diagnosis: bf16 at MT 1 costs exactly what fp32 costs (155 vs 157 us/expert at M 512, 557 vs 557 at 2048):
  fp32 itself is free, the cost is the 32-row sub-blocks it forced. Zones: gate/up and down compute per row tile
  identical at MT 1 / 2 (5.93 vs 11.76 us per zone), i.e. a fixed ~3.5-4 us per sub-block of pipeline overhead.
- Fix: gate/up row passes (`SE_GU_RP`, host `MIMO_FL_GU_RP` auto: dynamic counts, NP <= 2, NK_GU <= X_SLOTS): the
  pipeline keeps the 8-tile-sized sub-block, gate/up runs its K loop once per 4 / (2 NP) row tiles over the same x
  blocks (held in the x ring until the final pass; block b addressed with the signed wrap offset, first version
  without it gave rel err 0.41) and the same resident weights. K3: 50.2 / 126.2 / 419.7 us (fp32 was 49.1 / 156.5 /
  557.1, bf16 50.0 / 123.4 / 420.2). DSv4-F also gains (98.4 -> 89.2 at 512, 327.7 -> 284.3 at 2048); M3 neutral;
  K2 cannot use it (28 K-blocks > 24 x slots, no L1 for more).
- Tried, worse: full-sync fp32 DST (8 tiles, no math/pack overlap): K3 184 / 708 us at 512 / 2048.
- TP4: G 2 + 2 relay helpers is now the default for splits that would put gate/up on < 32 cores (G 1: 20.9 / 112.5 /
  435.0; G2: 24.9 / 91.8 / 346.1; G2 + 2 helpers: 24.7 / 85.2 / 309.1; G2 + 3 helpers: 25.3 / 82.0 / 305.0; G4 + 2
  helpers: 35.5 / 94.2 / 328.7). M-groups by rectangle (`MIMO_FL_GROUP_RECT`, auto for G 2: each relay multicasts
  only its group's rows, half the bytes) changed nothing (310.4). Profile at 2048: gate/up cores busy ~92%, 18.4 us
  median per 128-row sub-block vs 13.6 us minimum; the ~5 us stall is not the relays' tilize / DRAM / multicast
  bytes or the h chain (all probed): unresolved.

Final defaults (fp32 gate/up + row passes + TP4 G2), us per expert, 8 balanced experts, row-major bf16 DRAM input;
best existing = faster of unified / fused (fused cannot run clamped SiLU-GLU as the small-M path):
| shape | M 32 flat / existing | M 512 | M 2048 | norm ratio | PCC |
|---|---|---|---|---|---|
| K2 7168x2048 | 62.1 / 83.0 (1.34x) | 128.3 / 156.3 (1.22x) | 453.0 / 573.0 (1.26x) | 1.042 | 0.9982 |
| K3 3584x3072 situ | 50.4 / 74.0 (1.47x) | 125.7 / 149.4 (1.19x) | 419.7 / 544.2 (1.30x) | 1.051 | 0.9979 |
| K2 TP4 7168x512 | 24.4 / 36.4 (1.49x) | 85.5 / 90.1 (1.05x) | 310.1 / 286.3 (0.92x) | 1.025 | 0.9983 |
| DSv4-F 4096x2048 | 35.6 / 77.5 (2.18x) | 89.0 / 104.8 (1.18x) | 283.6 / 366.8 (1.29x) | 1.041 | 0.9981 |
| M3 6144x3072 oai | 80.6 / 111.7 (1.39x) | 185.7 / 200.1 (1.08x) | 666.8 / 736.5 (1.10x) | 1.049 | 0.9979 |
Existing ops' norm ratio on K2: 1.02-1.04.

### Correction: the relay "tilize ~80-96 cycles/tile" figure was not a tilize measurement
In se11_tz.cpp the `TZ_BLK` zone wraps `fast_tilize_block`, preceded by `TZ_IN` around `cb_wait_front(rm_cb)`.
`cb_wait_front` only blocks the UNPACK thread: the math / pack threads enter `TZ_BLK` immediately and wait inside it
for the unpacker, which waits for the reader's DRAM rows. So the TRISC1/2 `TZ_BLK` times (~2.3 us per 32 tiles)
include data-movement waits. The unpack thread (which waits correctly) shows ~0.57 us waiting + ~0.56 us issuing per
32-tile block (~23 cyc/tile issue). Documented BH fast tilize (PR #43577, gist cc166d487bbcea86d2bd311cfe934adf,
L1 to L1, one tile row): bf16 -> bf8 ~35-39 cyc/tile at widths 3-4, bf16 -> bf16 ~34-39 at widths 4-8.
Consequences: "relay tilize paces TP4 / K2 at large M" is UNSUPPORTED (it also fits the tiny gain from a third
helper); the ~5 us per sub-block gate/up stall on TP4 at M 2048 is unattributed again; the "tilize to bf16"
idea has no support (bf8 is not slower in the LLK data). Next: a tilize microbenchmark (pre-filled row-major CB,
idle core vs relay under load) and a per-sub-block critical-path trace across readers, relays, gate/up, down.

### TP2 vs existing (it was never compared at large M)
K2 TP2 7168x1024, us/expert M 32 / 512 / 2048: flat 34.4 / 115.4 / 438.0; unified 73.9 / 107.9 / 346.2; fused
48.3 / 133.7 / 458.1 -> flat LOST at 512 (0.93x) and 2048 (0.79x).

### Disjoint subgrids (`MIMO_FL_SG`, auto 2 for I <= 1024 with dynamic counts at helper-relay capacity)
Each subgrid is a complete copy of the pipeline on its own cores serving its own experts: gate/up rectangle k
(`MIMO_FL_SG_RECTS`, default 4x4 at cols 2-5 rows 0-3 and 2x8 at cols 8-9 rows 0-7: exactly the used cores, so
the freed cells become down cores), the 8 readers west of it, primary relay k + its helpers, half of the remaining
cores as down cores (split by distance to the rectangles), its own D_CHAINS chains / reader tails / coordinator /
go-done counts. Kernels unchanged except se_dyn.hpp `SE_SG`: every core assigns the active experts to subgrids
from the same counts (largest first onto the least-loaded subgrid, cost = sub-blocks + SE_SG_WCOST 2) and keeps its
own; an expert costing more than total / SE_SG is first cut by token range into SE_SG pieces (each streams its own
copy of the weights: cheap at low I). Host: subgrid-local indices (down core done slots, reader-tail columns / done
slots, pair-set of each reader), the subgrid id after the se_dyn args on every core, per-subgrid CT counts.
Result (us/expert, 8 experts; rag8 = [2000,40,90,60,30,300,20,50], spike8 = [3000 + 7 x 100], real8 =
[1196,88,715,300,64,40,500,20]):
| | uni32 | uni512 | uni2048 | rag8 | spike8 | real8 |
|---|---|---|---|---|---|---|
| TP4 1 subgrid | 24.4 | 85.5 | 309.4 | 59.9 | 78.8 | 65.2 |
| TP4 2 subgrids, no expert split | 21.8 | 63.4 | 225.0 | 61.4 | 87.1 | 53.1 |
| TP4 2 subgrids + split | 22.0 | 63.5 | 224.6 | 48.6 | 59.7 | 52.4 |
| TP4 best existing | 36.4 | 90.1 | 286.3 | | | |
| TP2 1 subgrid | 34.5 | 115.5 | 438.0 | 78.7 | 109.1 | 88.2 |
| TP2 2 subgrids + split | 39.0 | 112.0 | 396.9 | 80.9 | 111.7 | 89.3 |
| TP2 best existing | 48.3 | 107.9 | 346.2 | | | |
TP4 now beats the existing ops at every M (1.64x / 1.42x / 1.27x) and on ragged loads. TP2 still trails at large
M (0.87x at 2048, 0.96x at 512): a bf16 probe (MT 2: 392 us) shows it is not the sub-block size but gate/up compute,
32 gate/up cores in both modes (16 per subgrid x NP 2: 2 x 2 x 4 x 224 tile-matmuls per core per 128 rows, ~47 us
per subgrid, ~23.5 us overall). Needs 64 gate/up cores (32 per subgrid at NP 1): the second subgrid's reader-free
region is 3 x 10 = 30 cells, so that means two multicast rectangles per subgrid or 4 subgrids of 16 (NP 2, 4
readers each). Not done.
- Generalized to NSG rectangles (relay indexing: primary k = relays[k], rectangle k's helper j = relays[NR + NR j + k];
  generic placement for NSG >= 3: relays in the column west of each rectangle nearest its middle row, readers by
  nearest-rectangle round robin, readers no subgrid uses become down cores; down cores with > 8 columns run column
  passes of <= 8 in se6_dcompute (early pops only in the final pass)). The generic placement cost TP4 5-10% vs the
  measured 2-subgrid one (63.5 -> 68.4 at 512), so NSG 2 keeps the column-based reader split / old relay slots.
- TP2 options (us/expert M 32 / 512 / 2048, rag8 / spike8 / real8):
  | TP2 7168x1024 | 32 | 512 | 2048 | rag8 | spike8 | real8 | norm |
  |---|---|---|---|---|---|---|---|
  | 1 subgrid fp32 | 34.5 | 115.5 | 438.0 | 78.7 | 109.1 | 88.2 | 1.030 |
  | 2 subgrids fp32 (MT 1) | 39.0 | 112.0 | 396.9 | 80.9 | 111.7 | 89.3 | 1.030 |
  | 2 subgrids l1acc (MT 2) | 39.1 | 130.1 | 477.9 | 97.2 | 124.8 | 107.4 | 1.045 |
  | 3 subgrids fp32 (MT 1, 14 down x 15 cols) | 41.8 | 132.1 | 450.9 | 88.5 | 117.8 | 96.6 | 1.030 |
  | 3 subgrids bf16 (probe) | 41.8 | 106.2 | 369.9 | 74.5 | 102.2 | 90.5 | 1.259 |
  | **3 subgrids l1acc (auto)** | 42.1 | 108.3 | 373.2 | 81.1 | 104.1 | 90.5 | 1.045 |
  | best existing | 48.3 | 107.9 | 346.2 | | | | |
  Auto: I / 32 <= 16 -> 2 subgrids (fp32), <= 32 -> 3 subgrids (l1acc, rectangles 2:5:0:3, 2:5:4:7, 8:9:0:7).
  TP2 is now 1.15x at 32, ~1.0x at 512, still 0.93x at 2048 (was 0.79x). Remaining limit: 16 gate/up cores per
  subgrid at NP 2 (the grid's second reader-free region is 3 x 10, so 32-core rectangles for NP 1 do not fit twice).
  Regression K2 / K3 (1 subgrid) unchanged: 61.9 / 130.0 / 452.5 and 50.5 / 126.9 / 421.4.

### Dispatch cost (back-to-back launches, `MIMO_FL_B2B` / `MIMO_KREF_B2B`)
All earlier numbers were device kernel duration only. Back to back (no host sync between 8 launches), per launch
device span (last FW end - first FW start) / N minus the kernel duration, us; 8 experts, K2 unless noted:
| op | M 32 | M 512 | host enqueue per launch |
|---|---|---|---|
| flat, with the per-launch host copy resetting the relays' freed words | +77 (op-to-op 77) | +77 | ~30 |
| flat, relays zero their words in-kernel (`XMC_ZERO_WORDS`, no host copy) | +17 | +37 | ~30 |
| flat TP4 2 subgrids, host copy / in-kernel | +75 / +37 | +75 / +34 | ~31 |
| unified | +0.7 | +2.0 | ~6 |
| fused | +5.3 | +5.4 | ~6 |
The host copy was the whole op-to-op gap (77 -> 0.6 us) and would also have blocked tracing; the relays now zero
their freed words at kernel start (safe: the gate/up cores only bump them after consuming x the relay sent in the
same launch). What remains (~17-37 us) is the program's launch: ~19 kernel descriptors (several per role: readers
plain / reader-tail, forwarders, relay reader / tilizer / multicaster / one per helper, gate/up recv / compute,
down recv / weights / compute per column-width group) over fragmented core ranges, and unique runtime args on
~100 cores (gate/up / down cores carry 20-70 words of core coordinates). Not yet fixed; ideas: merge each role's
kernels into one binary with the role / width as runtime args (the SDXL diag-matmul fix: 16-19 -> ~1 us), fewer
per-core coordinate lists (derive from a base + stride), and measure under trace.

**Correction (dispatch cost is ~0.6 us, same as unified).** The "+17 / +37 us" above was a metric artifact: it spanned
from the 2nd launch's *FW start*, but on back-to-back launches some cores open the next launch's FW zone ~240 us
(M=512; ~112 at M=32) before the previous launch ends while they wait for it, so the span over-counted by 240/7 per
launch. Measured from the raw per-core device log instead (`b2b_launch_gap.py`: kernel-start-to-kernel-start
period and last-kernel-end -> next-first-kernel-start gap over 8 back-to-back launches; all cores start within
~1.2 us of each other):
| op | M 32 period / gap | M 512 period / gap |
|---|---|---|
| flat K2 | 496.8 / 0.60 | 1035.4 / 0.61 |
| flat TP4 (2 subgrids) | 175.0 / 0.47 | 506.1 / 0.50 |
| flat TP2 (3 subgrids + l1acc) | | 894.6 / 0.54 |
| unified | 1021.6 / 0.60 | 1185.5 / 0.58 |
| fused | 674.2 / 6.11 | 1508.4 / 6.12 |
(E8, K2, bf4 unless noted; unified/fused per `test_expert_ref_k2.py` defaults.) So the 13 kernels / 15 CBs /
~6.2K runtime-arg words (`MIMO_FL_SHOW_KERNELS=1` prints the anatomy: se5_recv 41 and se6_drecv 87 words per core
are the big ones) cost nothing measurable on device: the launch messages / binaries are staged while the previous
launch runs. Also tried: packing each kernel's cores into maximal rectangles (`_crs`, 257 -> 66 core ranges, keep
`MIMO_FL_CRS_SINGLE=1` for the old one-range-per-core) - no change (16.0 vs 16.1 on the flawed metric, gap
unchanged), kept since it is fewer dispatch writes. Real remaining cost is host-side only (~30 us enqueue vs ~6),
hidden behind >= 175 us kernels and removed entirely by trace. The earlier 77 us host-copy gap was real (op-to-op).

## In the MiMo layers (2026-09-27/28)

**Op builder.** The harness body moved to `tt/flat_expert.py` (`FlatExpert(device, weights, *, m, H, I, gids,
n_global, wdtype, act, pin, dyn, e2e)`; the harness only makes data / checks). Weights per device (a mesh: one
`MeshProgramDescriptor`, per-device global expert ids in the runtime args; lockstep allocation keeps every other
address equal). Per call: the kernels are cached by the buffer addresses (x, y, counts, regions, arena, words); the
CBs are rebuilt every call (they bind the launch's arena). On a `generic_op` program-cache hit the framework
re-applies every runtime arg and CB address from the call's descriptor (`apply_descriptor_runtime_args`).
* The arena (~1.2 MB/core on ~every core) and the relays' freed words are allocated per launch and freed after it: a
  persistent arena would clash with every other op's static CBs. Nothing in them outlives a launch (end to end the
  relays zero their words in-kernel) - checked with random garbage written to the arena's L1 between launches.
* The down coordinators' done words must start at zero (the coordinator leaves them zeroed): a 2 KB persistent L1
  tensor on the 1-3 coordinator cores.
* Host cost per call ~0.3 ms (allocs 0.07, descriptor 0.21, enqueue) vs >= 175 us kernels; the --profile build's
  Python path is much slower (b2b gap 811 us under Tracy), so measure host cost without it.

**64 experts per chip** (2x2 QuietBox: 256 / 4). SE_MAX_E 64 (meta page 2 KB) hung: the DM kernels' schedule struct
(SeDyn, 2 x 64-entry arrays) plus se_dyn_pin's full copy of it overflowed the 8 KB RISC local memory. Fixed: the pin
copy keeps only the original <= SE_MAX_E entries (off / cnt / subs / eid), subs is uint16, se_dyn_regions' arrays
uint8 (entries < SE_MAX_V <= 128, static_assert). Counts row: SE_DYN_HALF auto >= 4 x n_global (1 KB for 256).

**Model wiring.** `MIMO_FLAT_EXPERT=1` (tt/ffn.py): TtMoE builds one FlatExpert over the mesh (device (r, c) gets the
EP table's experts table[c, r], the unified op's global ids), feeds it the dispatch's row-major bf16 buffer and the
routing's [1, 256] counts / regions rows directly (no tilize), skips the unified weights. Real routing on the
decoder-layer test (random prompt embeddings) is very spiky: per chip one or more experts at the full capacity
(2048 = 2 x 1024 tokens), most of the 64 empty; standalone replays of each chip's counts: 376 - 1266 us.

**The hang: a nonzero write transaction ID left in the NoC command buffer.** The decoder layer passed its first MoE
call and hung on the second, all 4 chips, the whole pipeline waiting (gate/up on GO, coordinator on done words, compute
backpressured inside matmul / tilize LLKs). Minimal repro: flat -> one 1024x4096x4096 matmul -> flat, single chip, no
fabric, any mode (even pre-tiled static); flat back to back (50x), an eltwise add or a 1-core matmul between: fine.
Ruled out: routing counts, in-flight work (sync before the call), arena address moves, arena contents (zeros /
garbage fills), done-word corruption (probed zero). The watcher named it: "BRISC detected invalid NOC command buffer
state before starting the next kernel (write-capable NOC packet tags must be zero so implicit transaction ID users
start with transaction ID 0). Current kernel: se6_drecv.cpp". se6_drecv / se5_recv send h / x with
`noc_async_write_one_packet_with_trid` and exited with their last ID in the write command buffer; the next program's
plain writes then carried it and its write barriers / flush checks hung (back to back the next program was the flat
op itself, whose trid users set their own IDs). se10_fwd / se9_rdown already reset theirs. Fixed: both reset to ID 0
before exit, and every flat DM kernel now ends with `noc_async_full_barrier()` (the relay multicaster also exited
without an atomic barrier). test_flat_expert_mesh.py (2x2, spiky counts, a matmul + SDPA + all-gather and a pinned L1
buffer between launches) guards it; the harness has `MIMO_FL_BETWEEN=matmul`.

Decoder layer L1 (SWA + MoE) with the flat expert vs HF: PCC 0.99956 (unified: 0.9996).

**Layer perf, flat vs unified** (2x2 QuietBox, real prompt routing, 33k ctx; `test_layer_perf.py`). Device kernel
time (sum of per-op max over chips, --profile) and unprofiled wall time per layer (`MIMO_PERF_WALL=6`, median of the
warm iterations; host + device):
| layer | tok/chip | kernel unified -> flat | expert op: tilize + unified -> flat | wall unified -> flat |
|---|---|---|---|---|
| L1 SWA + MoE | 640 | 6059 -> 4609 us (-24%) | 2247 -> 1104 us (2.04x) | 5.77 -> 5.13 ms (-11%) |
| L1 SWA + MoE | 2048 | 13228 -> 11414 us (-14%) | 3643 -> 2097 us (1.74x) | 11.73 -> 10.29 ms (-12%) |
| L5 GA + MoE | 640 | 10242 -> 7882 us (-23%) | 4180 -> 1915 us (2.18x) | 10.21 -> 7.92 ms (-22%) |
| L5 GA + MoE | 2048 | 21609 -> 18450 us (-15%) | 5543 -> 2739 us (2.02x) | 20.61 -> 17.83 ms (-13%) |
(Reduce-scatter varies +-300 us run to run in both modes.) Host cost per flat call on the mesh without the profiler:
0.84 ms (descriptor 0.29 with the per-device descriptors cached and only the CBs swapped, enqueue 0.35, allocs), a
kernel-cache miss (new buffer addresses: once after the warm-up iteration) 29 ms; under Tracy both are ~10x (a miss
340-965 ms), which is why profiled layer spans looked like 180-500 ms. L1 at 640 tok/chip gains 0.64 ms wall of the
1.14 ms expert saving: that layer is partly host-bound, the flat call's host path shows; L5 (longer attention) hides it.

## C++ op: ttnn.experimental.deepseek_prefill.flat_routed_expert (2026-09-28)

The model path (dynamic counts, row-major x, >= 256 tokens per expert: helper relays, pinning, gate/up split,
1-3 subgrids, reader tails) as a C++ device op (`ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/
flat_routed_expert`): `device/flat_routed_expert_plan.cpp` (the layout, ported from FlatExpert: role cores, split,
chains, arena; Python's `min()` first-minimum and round-half-even kept), `device/flat_routed_expert_program_factory.cpp`
(classic factory: every buffer-address runtime-arg word is recorded as a patch (kernel, core, arg, source tensor,
offset); `override_runtime_arguments` re-applies them and the arena-backed CB addresses on a program-cache hit),
the kernels moved there (one copy; stubs at the old research paths). The expert ids now come from a per-device table
the kernels read (se_dyn.hpp `SE_DYN_NARGS` = 7: counts, regions, row bytes, band lo / hi, id table, its bytes): one
program for every device. The host wrapper allocates y and the per-launch arena / words, launches, frees the scratch;
`flat_routed_expert_plan` (ttnn._ttnn.operations.experimental) gives Python the plan (weight layout, coordinator
cores); `tt/flat_expert.py` FlatRoutedExpert = plan + `flat_weight_regions` (the weight layout, shared with the Python
builder) + the op. `MIMO_FLAT_EXPERT=1` uses it in TtMoE (`=py`: the Python builder).

Checks: `tests/unit/test_flat_routed_expert_op.py` (MiMo 64 experts / K2 reader tails / TP4 2 subgrids / TP2 3 subgrids
+ l1acc): C++ plan == Python plan, y bit-identical to the Python builder over two launches (the second after a moved
arena: the cache-hit patch), min PCC 0.998 vs the quantized reference; test_flat_expert_mesh.py (2x2, other ops
between launches) with either implementation; decoder layers L1 / L5 PCC 0.99956 / 0.99196.
Host cost per call on the 2x2 mesh: 0.154 ms enqueue, 0.28 ms after an address move (Python builder: 0.84 ms, 29 ms).
Wall per layer (unprofiled, median, 33k ctx): unified -> flat Python -> flat C++:
| layer | tok/chip | unified | flat py | flat C++ |
|---|---|---|---|---|
| L1 SWA + MoE | 640 | 5.77 ms | 5.13 | 4.94 (-14%) |
| L1 SWA + MoE | 2048 | 11.73 | 10.29 | 10.32 (-12%) |
| L5 GA + MoE | 640 | 10.21 | 7.92 | 7.88 (-23%) |
| L5 GA + MoE | 2048 | 20.61 | 17.83 | 17.83 (-13%) |

**The layer traced** (`MIMO_TRACE_REGION=200000000 MIMO_PERF_TRACE=20`: capture one layer call on a persistent x,
replay; traced output bit-identical to eager, checked). ms per layer, eager wall / traced back to back:
| layer | tok/chip | unified eager | unified traced | flat C++ eager | flat C++ traced | traced gain |
|---|---|---|---|---|---|---|
| L1 SWA + MoE | 640 | 5.75 | 4.98 | 4.92 | 4.10 | -18% |
| L1 SWA + MoE | 2048 | 11.71 | 11.62 | 10.29 | 10.17 | -12% |
| L5 GA + MoE | 640 | 10.21 | 9.85 | 7.90 | 7.56 | -23% |
| L5 GA + MoE | 2048 | 20.63 | 20.50 | 17.82 | 17.70 | -14% |
Device time (profiler: sum of per-op kernel durations, the slowest chip per op) excludes op-to-op gaps and host;
wall adds them; traced removes the host, so traced ~ device execution. At 640 tok/chip eager layers are partly
host-bound (0.3-0.8 ms of eager wall is host); at 2048 they are device-bound (traced ~ eager).

**Device time vs traced, 0 starting context** (`MIMO_PERF_CTX=8192 MIMO_PERF_KV_ACTUAL=0`: the first chunk; the cache
must be longer than one chunk, the ring SDPA's chunked path is the only one sliding-window supports) with
`TT_METAL_SHM_TRACKING_DISABLED=1 LOGURU_LEVEL=ERROR` (#55960's settings). `analyze_device_time.py`: per-op max (sum of
each op's slowest device: over-counts, a collective's kernel time includes waiting for peers), busy (per device, sum
of its kernels), span. ms per layer:
| layer | tok/chip | impl | per-op max | busy (slowest device) | traced | eager |
|---|---|---|---|---|---|---|
| L1 SWA+MoE | 640 | unified | 5.41 | 4.65 | 4.67 | 4.84 |
| | | flat | 4.76 | 3.98 | 4.01 | 4.17 |
| L1 SWA+MoE | 2048 | unified | 12.56 | 11.16 | 11.20 | 11.30 |
| | | flat | 11.48 | 10.07 | 10.07 | 10.16 |
| L5 GA+MoE | 640 | unified | 8.23 | 7.57 | 7.57 | 7.77 |
| | | flat | 5.49 | 5.15 | 5.16 | 5.37 |
| L5 GA+MoE | 2048 | unified | 14.72 | 13.79 | 13.78 | 13.89 |
| | | flat | 11.59 | 10.87 | 10.88 | 10.97 |
Traced == the slowest device's busy time (within 0.03 ms): traced runs are device-bound. The env settings bring eager
within 0.1-0.2 ms of traced (at 32k ctx without them the gap was 0.3-0.8 ms at 640 tok/chip).

## Precision stress: unified / fused / flat (2026-09-28)

`tests/perf/test_expert_precision.py` (one chip, H 4096, I 2048, 8 experts, capacity 512, ragged counts
512 / 1 / 31 / 33 / 0 / 200 / 64 / 300; unified + fused get tiled bfp8 x, flat row-major bf16), 13 x distributions x 3
weight sets, vs a quantized reference (bfp8 x, bfp4 W, fp32 math) and fp32. Raw: `expert_precision_results.tsv`.
* No NaN / Inf anywhere; every op scale-invariant (x 0.01 .. 100, rows 1e-3 .. 1e3, one-hot, sparse, heavy tails,
  outlier channels, constant rows).
* Arithmetic error vs the quantized reference (rel L2, normal / large weights): unified 1.6-3.5%, fused 2.4-4.5%,
  flat 6.7-7.5%; norm ratio unified 1.00-1.02, fused 1.00-1.03, flat 1.03-1.04: flat's error is ~2x unified's and
  systematic (a norm gain, the same on every input) -> the down projection's bf16 DEST accumulation over K = I
  (ties-away rounding, measured 1.023 at K 2048) is the likely source (gate / up already run fp32 DEST). Unverified.
* End to end (vs fp32 x / W) all three ~equal: rel 20% with N(0, 0.02) weights (the bfp4 weights dominate).
* All-positive weights + all-positive x (abs normal, uniform, one-hot, constant): outputs are nearly a common mode, so
  PCC (mean-removed) drops to 0.81-0.99 for every op while rel error stays 0.5-3.3% (unified < fused < flat); with
  positive weights and zero-mean x, a few near-cancelling rows have large per-row relative error (flat worst 11x at
  x100, unified 1.1x): small absolute errors on tiny rows.

**Equal inputs** (all three ops get the row-major bf16 dispatch buffer; `expert_precision_results.tsv` now holds that
run): unified / fused worst-case arithmetic error rises 3.5 -> 4.5% / 4.5 -> 5.0% (device bfp8 packing of x costs
~1% over host quantization; flat always paid it), flat unchanged 6.7-7.5%. Host estimate of the remaining parts:
h stored bfp8 ~1.1% rel, y bfp8 0.7%.

**Accumulation sweep** (flat, Python builder: `MIMO_FL_GU_ACC` x `MIMO_FL_DN_ACC`; accuracy on 5 x cases with normal
weights, `expert_precision_acc_sweep.tsv`; kernel time MiMo 64 experts real counts / 8 x 512):
| gate/up | down | q rel | q norm | MiMo 64 | 8 x 512 |
|---|---|---|---|---|---|
| fp32 | bf16 (default) | 6.7-7.5% | 1.030-1.041 | 1282 us | 713 us |
| fp32 | fp32 half-sync (4-tile DST: 2 K loops per row) | 5.5-6.0% | 1.003-1.014 | 1771 | 937 |
| fp32 | fp32 full-sync (8 fp32 tiles, `fp32full`) | 5.5-6.0% | 1.003-1.014 | 1302 (+1.6%) | 725 (+1.7%) |
| l1acc | bf16 | 6.8-8.4% | 1.032-1.054 | 1341 | 719 |
| l1acc | fp32 half-sync | 5.5-6.6% | 1.006-1.028 | 1767 | 938 |
| bf16 | bf16 / fp32 | up to 19% | up to 1.17 | fails the harness gate | |
fp32 gate/up stays the best (more accurate than l1acc and faster); fp32 down in a full-sync DST removes the down
bias for ~1.7% (the pack is tiny next to the K loop, so losing math/pack overlap is cheap).

**bf16 h** (`MIMO_FL_H_BF16=1`: 2 KB h tiles; at MiMo's shape only 2 h buffers and a 1.5-expert down ring fit L1, which
also drops the pinned down regions): MiMo 64 experts 2085 us (vs 1302 with bfp8 h, fp32 full-sync down), 8 x 512 1100 us
(vs 725); arithmetic error 5.96 -> 5.76% (vs the old, mismatched reference). The control (bfp8 h with the same smaller
ring / 2 h buffers) runs 1640 / 863 us: half the +60% is the L1 compromise, half the doubled h traffic. Not worth it.
HiFi2 matmuls: bit-identical accuracy (bfp4 x bfp8 products are exact in LoFi), MiMo 64 2247 us. Not the source either.

**The reference was mismatched.** From fp32 weights the flat upload rounds through bf16 before bfp4 (its weights ==
bfp4(bf16(W)) exactly, `test_flat_weight_quant.py`); the reference / unified quantize fp32 -> bfp4 directly: ~0.75% of
the elements one bfp4 step apart (equally close to fp32: 11.28% vs 11.27%), charged to flat as ~2.5% "arithmetic
error". The model's weights are bf16 already (no difference there). `test_expert_precision.py` now draws bf16-valued
weights. Fair arithmetic error (vs the quantized reference, equal row-major inputs, normal weights, 13 x cases):
| op | q rel | q norm |
|---|---|---|
| unified | 1.6-4.5% | 0.996-1.032 |
| fused | 2.4-5.0% | 1.000-1.036 |
| flat (fp32 gate/up, bf16 down: the default) | 3.9-5.2% | 1.030-1.042 |
| flat, fp32 full-sync down | **1.1-2.5%** | **1.004-1.016** |
With fp32 full-sync down (+1.7% time) flat is the most accurate of the three on every input. End to end (vs fp32) all
~19.6-21% (the bfp4 weights). No NaN / Inf anywhere.

**fp32 full-sync down across M and TP shapes** (8 experts balanced, row-major x, pinned; kernel us, bf16 -> fp32full):
| shape | M 32 | M 512 | M 2048 |
|---|---|---|---|
| MiMo 4096 x 2048 | 270.6 -> 271.3 (+0.3%) | 721.8 -> 718.3 (-0.5%) | 2297.8 -> 2333.6 (+1.6%) |
| K2 7168 x 2048 (reader tails) | 472.6 -> 472.2 | 1051.3 -> 1060.0 (+0.8%) | 3636.5 -> 3634.9 |
| K2 TP2 7168 x 1024 (3 subgrids, l1acc gate/up) | 263.8 -> 264.2 | 894.1 -> 881.2 | 2989.3 -> 2969.8 |
| K2 TP4 7168 x 512 (2 subgrids) | 208.8 -> 208.6 | 509.6 -> 507.2 | 1799.9 -> 1796.3 |
At most +1.6% (MiMo compute-bound, 5 down columns per core), within noise elsewhere; every run passes the harness's
norm / PCC checks. Cheap enough to be the default.

**Math utilization at large M** (8 experts balanced, fp32 full-sync down; FLOPs = 6 M H I E; peak = 110 cores x 4096
FLOP/cycle x 1.35 GHz = 608 TFLOP/s):
| shape | M 2048 | M 4096 |
|---|---|---|
| MiMo 4096 x 2048 | 2333.6 us, 359 TF/s (59%) | 4469.3 us, 369 TF/s (62%) (bf16 down 4382.8) |
| K2 7168 x 2048 | 3634.9 us, 397 TF/s (65%) | 7090.7 us, 407 TF/s (67%) |
| K2 TP2 7168 x 1024 | 2969.8 us, 243 TF/s (40%) | 5790.4 us, 249 TF/s (41%) |
| K2 TP4 7168 x 512 | 1796.3 us, 201 TF/s (33%) | 3515.1 us, 205 TF/s (34%) |
Doubling M adds only 1-3 points: the op is already in its steady state at 2K. Zone profile at 4K
(`analyze_role_busy.py`, E=2; the profiler buffer overflows, so lower bounds): MiMo gate/up MM busy 91%, down 86%, the
gate/up matmul itself ~73% of LoFi peak inside its zone (both roles near-saturated, so the rest is matmul efficiency).
TP4: the 32 gate/up cores 94.5% busy, the 72 down cores wait for h 80% of the time (gate/up-bound: I/4 gives down 4x
less work, but the split still reserves down cores for H = 7168 output columns).
Tried at M 4096 (all worse than the defaults): TP4 1 subgrid G4 (64 gate/up cores) 5201.7 us, 1 subgrid G2 4875.1,
3 subgrids (48 gate/up cores) 4258.9 (vs 3515.1); MiMo l1acc gate/up 4975.3 (vs 4469.3); MiMo reader-tail down
(`MIMO_FL_RDOWN=1`) fails the reader-down tensor bounds check at MiMo's shape (not pursued).

## MoE block: all-gather architecture (2026-09-28)

`MIMO_MOE_AG=1` (implies the flat expert; tt/moe_ag.py, kernels tt/kernels/moe_ag/, tests tests/perf/test_moe_ag.py,
test_hbw_ag_probe.py): no dispatch / combine. Per MoE layer on the 2x2:
x TILE -> row major -> `ttnn.experimental.high_bw_all_gather` over the rows (x, top-k indices, weights; persistent
outputs) -> RoutePlan (generic_op, 64 cores: counts / regions / token_index / y_slot on device, trace safe) -> flat
expert, indexed (token_index) -> UntilizeActive (bfp8 y -> bf16 rows, only tiles holding tokens) -> LocalReduce
phase 1 (the other row's tokens: sum_k w * y) -> high_bw_all_gather over the rows -> LocalReduce phase 2 (this row's
tokens + the peer's partial as a weight-1 pair) -> high_bw_all_gather over the cols -> AddRowsTiled (col sum +
tilize) -> [1, 1, S, H] TILE replicated over TP. Decoder layers vs HF: L1 0.99957, L5 0.99197 (dispatch path 0.99956
/ 0.99196).

**high_bw_all_gather on our FABRIC_2D 2x2** works without fabric changes (no L1_SMALL: it warns and puts its
semaphores in L1; it synchronizes only on a program-cache miss). Over the rows, us (4 links; it scales with links,
1 / 2 links are slower than ttnn.all_gather):
| | 640 tok/chip | 2048 |
|---|---|---|
| x RM 8 KB pages: hbw / ttnn.all_gather | 71 / 105 | 192 / 224 |
| x TILE: hbw / ttnn.all_gather | 57 / 73 | 153 / 218 |
| x RM 2 KB pages ([S * 4, 1024]) hbw | 58 | 153 |
| top-k idx or weights [S, 8] 16-bit: hbw / all_gather | 15 / 29 | 19-21 / 71 |

**Pieces** (us, uniform routing, one chip): RoutePlan 29 / 53 (first version 93 / 120: an O(cores x experts)
L1 loop per core; now expert cores do the column prefix, go signals by multicast); ttnn.to_layout(y) 255 / 637 vs
UntilizeActive 200 / 457 (bfp8 untilize is compute bound, ~1300-1800 cycles per tile in both; the real fix is
row-major bf16 y from the expert); LocalReduce 101 / 297 (~370 GB/s DRAM); fused send-back phase 1 + 2 = 132 / 379 vs
reduce + exchange add 147 / 439; AddRowsTiled 60 / 144 vs add + ttnn tilize 116 / 292; UntilizeX (2 KB pages) 47 / 120
vs to_layout 27 / 92 (`MIMO_MOE_AG_XPPR=4` knob only; default one 8 KB page per token row).

**Layer device time** (busy, slowest chip, 0 ctx; first all-gather version: to_layout untilize, separate adds,
ttnn tilize), ms: L1 640 3.98 -> 2.84, L1 2048 10.07 -> 7.10, L5 640 5.15 -> 4.08, L5 2048 10.87 -> 8.49.

**After fused send-back + AddRowsTiled + UntilizeActive** (busy, slowest chip, 0 ctx, ms; dispatch path -> now):
L1 640 3.98 -> 2.70 (-32%), L1 2048 10.07 -> 6.85 (-32%), L5 640 5.15 -> 3.93 (-24%), L5 2048 10.87 -> 8.17 (-25%).
At 2048 the slowest chip's MoE data movement: x untilize 92, gathers 189 + 33, plan 60, UntilizeActive 628 (real
routing loads chip 2), reduce phase 1 / 2 214 / 304, gathers 192 / 187 (+ waits for the slowest expert), col add 143.

**Galaxy generalisation (proxy meshes, test_moe_ag_mesh.py, random routing / y, block only):**
- rows = 1 (1x4): no gather; rows = 2: fused fast path; rows > 2 (4x1): LocalReduce writes the [T, H] partials as
  bf16 tiles (tilize fused: 276 vs 201 + 161 us at 4x1 S640) -> ttnn.reduce_scatter over the rows (row-major RS is a
  slice / tilize / concat composite: 1287 vs 822 us, not used).
- TP over > 2 cols: `MIMO_MOE_AG_TP=rsag` default (ttnn RS + AG on tiles: 1x4 162 / 284 us vs hbw gather + 4-way
  SumBlocksTiled 220 / 404 at S 640 / 1280); 2 cols keep hbw gather + AddRowsTiled.
- PCC of the final sums vs host >= 0.999993 on 2x2 / 4x1 / 1x4; EPC 8 (Galaxy count) at T 5120 and 16384 on 4x1:
  0.999998 (RoutePlan 48 / 127 us; 8x8 cores, lists up to T entries in L1).
- 4x1 EPC 8, T 5120 (= Galaxy chunk 5120): gather 227 (ring, 4 links), plan 48, untilize 86, LocalReduce 282 (writes
  the dense [T, H]), reduce-scatter 284 us. The > 2-row send-back moves the dense partials although a chip touches
  only ~22% of the tokens on a Galaxy: estimated Galaxy (8 rows, 2 links) ~0.5-0.7 ms gather + ~0.65 ms RS per layer,
  i.e. no better than dispatch / combine (~0.9-1.2 ms by the per-entry model) -- the Galaxy needs a sparse send-back.
- Untestable here: 8-row fabric (2 links, torus or line), 4-col TP at 2 links, hbw on the Galaxy's FABRIC_2D_TORUS_XY,
  T 5120 with the real 32-chip expert placement in the model (the flat expert with EPC 8).

## Row-major bf16 y (2026-09-29): no MoE-output untilize

`flat_routed_expert(..., y_row_major=True)` (FlatRoutedExpertConfig.y_row_major, part of the program key; the plan is
unchanged): y is [rows, H] bf16 ROW_MAJOR, one 8 KB page per flat row. Down compute (se6_dcompute.cpp, also on the
reader tails) pack-untilizes each row tile (`pack_untilize_dest<cpw, pcd>`; in this mode the column-pass width is the
largest divisor of PCD that fits DST, so any PCD works) and pushes it as PCD pages; the writers (se6_drecv.cpp,
se9_rdown.cpp via se_yrm.hpp) send one PCD * 64-byte segment per token row, several row tiles in flight on write
transaction ids 8.. (ids 1..3 are the h links), retired in order, write trid reset to 0 at exit. The out CB (Float16_b)
reuses the arena's out region, as many bf16 row tiles as the arena holds (<= 2 MT, <= 8). Default stays bfp8 TILE.
MiMo: moe_ag uses it by default (MIMO_MOE_AG_YRM=0: bfp8 + UntilizeActive, built lazily); LocalReduce reads y directly;
the 151 MB (2048 tok/chip) untilized-y buffer is gone.

Checks: tests/unit/test_flat_expert_yrm.py (MiMo / K2 reader tails / TP4 / TP2, 2 launches incl. moved arena): row-major
y within 1.2% of the 16-value block max of the bfp8 y (= bfp8 rounding), PCC >= 0.998 vs the quantized reference.
test_flat_routed_expert_op.py 4/4 (needed CT 7/8 in the Python builder's se11_xrd args, from the x_pages_per_row change).
Decoder layers: L0 0.99998531 (same), L1 0.99956720 -> 0.99958960, L5 0.99196511 -> 0.99202095 (y no longer bfp8).
Loop test R6 @2048: worst hidden 0.99931, K 0.99924, V 0.99759 (unchanged).

Expert alone (test_flat_expert_indexed.py, one chip, 64 experts): row-major y costs +1.6..5.2% vs bfp8 y (writes in
flight or a deeper out CB made no difference once pipelined; first, serial per-row-tile version was +8..12%): plausibly
the 2x y write bytes against the weight stream (largest on uniform routing, the most weight-bound).
Layer (8K ctx, slowest chip busy, bfp8+UntilizeActive -> row-major y):
| layer | tok/chip | layer | expert | MoE ops after the expert |
|---|---|---|---|---|
| L1 SWA+MoE | 640 | 2.77 -> 2.56 ms | 989 -> 997 us | 686 -> 466 us |
| L5 GA+MoE | 640 | 4.40 -> 4.25 ms | 1801 -> 1923 us | 696 -> 440 us |
| L1 SWA+MoE | 2048 | 6.81 -> 6.26 ms | 2075 -> 2123 us | 1782 -> 1195 us |
| L5 GA+MoE | 2048 | 8.99 -> 8.56 ms | 2588 -> 2646 us | 1727 -> 1245 us |

## In the K3 token-split MoE block (2026-10-16/17, bh-lb-09 2x4, 12x10 grid, 200G)
The flat expert runs twice per K3 block (tt_k3_token_split_moe.py): the routed experts fused with combine_rf
(flat_routed_expert_with_program, L1 hand-off, 56 experts / chip, ~91 tokens each, bf4 LoFi) and the shared expert
(3 parts of I 2048 over the chip's 320 tokens, bf8 HiFi2, bf16 x / y). Block-level history and tables:
FLAT_COMBINE_OVERLAP_WORKLOG.md ("K3 block ..." sections). The October flat-expert entries 8e-8g (DRISC gate/up
senders, the x-read isolation, K3 / 4x4 routing) are only in the tt-metal-pr58093 worktree's copy of that worklog.

Routed experts + combine, traced (experts program, us): 2433 -> 2372 with x read once (MIMO_FL_X1, set for this
program only - read when the program is built). Floor at 512 GB/s ~2070 (1059 MB / chip / call).
- DRAM bound at ~87%: with no weight traffic (W_NOREAD + W_NOFWD) the program takes 1221 (compute + x + h + combine);
  the readers' gate/up forwarding alone (no DRAM) ~1990 (694 MB, ~350 GB/s); all DRAM reads, no forwarding 2430; the
  gate/up reads alone stream at 93%, the full mix (gate/up + down + x) at 86-87%.
- Weight source does not matter: dual / pure DRISC, DRISC tiles 1 / 2, DRISC forced (MAXCNT 1024): 2362-2380. K3's
  busiest expert exceeds the pure mode's 96-token switch, so calls normally take the Tensix-reader gate/up path.
  Down weights via the DRISCs: ruled out (the forwarding path would need ~2980 us for 1041 MB).
- No effect: GDDR arbitration weights (MIMO_FL_DRISC_MPFE, 6 settings), combine release (CMB_EARLY), h transfer
  (H_NODATA), MIMO_FL_DRISC_MAXCNT 160. Worse: X2 2659, XFWD 2859, NH 2 / 3 2471 / 2457, X1 + NH 2 2396.
- Probe caveats on this path: W_NOREAD skips only DRAM reads (blocks are still forwarded; W_NOFWD stops that).
  MIMO_FL_NO_MATH hangs (the gate/up side never publishes h; se6_dcompute waits in cb_wait_front(h_all_cb)) - broken.
Shared expert (3 x 320-row parts): ~525-530 us, compute / pipeline bound (~160 TFLOP/s whatever the fidelity: LoFi
510; bfp8 x, DRISC on / off, 4 parts no better); X1 hurts it (523 -> 702: one primary's tilize + double multicast caps
320-row parts). Dense ttnn at TP shapes beats it (TP4 gate linear ~245 TFLOP/s at M 1280): zz_scratch_k3_se_tp.py.
Next candidates: the down weights' bank layout / access order (the mix's 87% vs 93%), fixing NO_MATH for a clean
traffic-only measurement.

### K3 routed experts: down-weight read layout / schedule (2026-10-17, traced, X1 on; experts / block us, base ~2375 / 3645)
| knob | experts | block |
|---|---|---|
| MIMO_FL_DW_ROT=0 (one bank per down core) | 2565 | 3851 |
| MIMO_FL_DRING=1.5 (down ring, experts) | 2365 | 3654 |
| MIMO_FL_DRING=3 | plan TT_FATAL (out region / L1) | - |
| MIMO_FL_PIN_RATIO=0 (always pin) | 2438 | 3723 |
| MIMO_FL_PIN_RATIO=1000 | 2376 | 3644 |
The defaults are right for K3: bank rotation of the down weights is worth ~190 us here (more than at K2.7 / GLM, worklog
comment in the plan: 55% -> 65% alone); ring depth and pinning ratio change nothing or hurt. With the knobs exhausted
the remaining ~13% (2375 vs ~2070 at 512 GB/s) needs a DRAM / NoC trace of the mix (tt-npe or the device NoC tracer)
or a fixed MIMO_FL_NO_MATH for a clean traffic-only time.

### K3 routed experts + combine under tt-npe and the per-core profiler (2026-10-17)
NoC trace (python -m tracy -r -p --collect-noc-traces on scratch_flat_combine_rf.py, FC_EPC 12 FC_S 64 (~85 tokens /
expert, K3-like), X1, TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT 4000; tt-npe kept off PYTHONPATH during collection - its
automatic all-trace analysis exhausted host memory on the first try - then run by hand on one program / chip):
complete (238 MB of DRAM reads vs ~230 expected; max 3963 events / RISC; se9_rdown's reads are present now).
- tt-npe (P150 model, the run's soc descriptor): estimate 1.06 M cycles = 786 us vs 788 traced; congestion impact 0%,
  link util avg 8% / max 35% (demand bursts to 170% cost nothing), DRAM util 70% over the whole span. Not NoC bound.
- DRAM read rate in 20 us bins: ~400-500 GB/s (avg ~445, 87%) in steady state, then a drain: last gate/up read at 553
  us, down 587, x 615, kernels p50 595 / p90 782, end 788. Traffic: gate/up 54 KB reads (BRISC), down 9 KB (NCRISC),
  x 1 KB; each bank serves ~29 MB through one NoC endpoint and ~1 MB through the other. Down reads moved to the other
  endpoint (MIMO_FL_DN_NOC0=1): worse (experts 2452 vs 2372, block 3792 vs 3649 - NOC1 DRAM reads are slower here).
Full size, per-core kernel end times (device profiler), chip 0:
| program | cores | bulk end (p50-p75) | last core |
| flat alone (standalone, per-call sync) | 94 | 2386-2400 | 2407-2411 |
| combine alone | 21 | 921-929 | 935-943 |
| fused (standalone, per-call sync) | 115 | 2480-2500 | 2628-2685 |
| fused, inside the K3 block trace (profiled) | 115 | 2305-2340 | ~2440 |
So the fused program = the flat cores (slowed ~4% by the combine's traffic) + ~110-120 us of combine running alone
after the experts (the last experts' rows: fabric hops, home accumulate, final writes). The drain is ~1/3 of the gap to
the DRAM floor; the rest is the steady-state DRAM efficiency (87%).
Follow-up (same day): the combine tail varies run to run - a second profiled trace (TT_METAL_PROFILER_SYNC=1) gives
per-chip experts programs of 2357-2376 us (the unprofiled traced 2372) with tails of 17-51 us (chip 0 44-51), vs
22-115 in the first. Cross-chip alignment from sync_device_info.csv (per-device frequency ratio + shift vs device 0)
did not work either way round (offsets of seconds), so "start skew vs a fabric hot spot" is unresolved. Profile-log
parsing note: traced replays reuse the run host id - key program instances by (device, run id, trace id, trace id
counter). Conclusion: the tail is <= ~1/3 of the gap and noisy; the steady-state DRAM efficiency (87%) is the bulk.

### K3 routed experts: the DRAM read mix alone (zz_scratch_dram_mix.py, 2026-10-17)
Plain NoC reads (transaction-id queues, no compute) replaying the K3 plan's weight streams per call on one chip, on
the plan's own cores (flat_routed_expert_plan dumped at K3 shapes on the 12 x 10 grid: 16 gate/up readers in logical
columns 0 / 1 / 7 / 8 next to the DRAM tiles, 28 down cores of 4 output tile columns, down blocks 4 x 4 bf4 = 9 KB):
| pattern (DRAM in flight: gate/up 4, down 2 per core) | us | % of 512 GB/s |
| gate/up alone (54 KB, own bank) | 1356 | 99.9 |
| down alone (9 KB, rotating; depth 2 / 8) | 1044-1067 | 64-65 |
| down alone 36 KB rotating / 9 KB pinned per bank | 1130 / 1108 | 60 / 54 |
| down alone from 24 cores next to DRAM | 958-970 | 70-71 |
| mix as today (depth 2 / 4) | 2256 / 2230 | 90.1 / 91.1 |
| mix, down 36 KB / pinned | 2285-2302 / 2197-2202 | ~89 / ~89 |
| mix, down from cores next to DRAM, depth 2 / 8 | 2300 / 2153 | 88.3 / 94.4 |
(A first version with barrier-batched reads on arbitrary cores gave gate/up alone 62-67%: reader placement / queueing
matter - the gate/up readers' one-hop placement is what gets them to 100%.) Reading: today's pattern bounds the
experts at ~2230-2256 us of DRAM alone; the real program (2372) adds ~120-140 (x reads, compute coupling, the combine
tail). The down stream is the weak part (~65% alone, whatever the block size, depth or bank pinning); bigger or
pinned blocks do not help the mix; only down reads issued next to the DRAM columns with deeper queues lift it (+3%,
~70-80 us) - i.e. down weights read by near-DRAM cores and forwarded to the down cores.

### K3 routed experts: three more ideas (2026-10-17, traced; base experts ~2372, block 3640-3650)
1. Down-weight blocks per read barrier (MIMO_FL_DW_BATCH, now an env knob, default DW_BATCH 2): 4 / 6 / 8 -> experts
   2354 / 2410 / 2448; batch 4 for the routed program only (K3_EXP_DW_BATCH): block 3651 / 3651 - noise. Off.
2. Combine slot order (MIMO_FL_CMB_SLOT_ORDER): experts 2374, block 3650 - no change.
3. Reader down columns (MIMO_FL_RDOWN=1, the near-DRAM readers take down columns - the direction the DRAM-mix
   benchmark favours): crashes with the env on both flat ops (core_coord TT_THROW); scoped to the routed program
   (K3_EXP_ENV=..., new: comma-separated flat knobs for that program only) it overflows the fused program's arena
   (factory.cpp:1051 off + size <= arena.aligned_size_per_bank()) - needs the arena / combine L1 layout resized.

### DeepSeek V4.1-Flash shapes (2026-10-17): H 5120, I 2304, 384 experts, top-6, clamped SwiGLU (limit 10)
Balanced sweep (test_flat_balanced_sweep.py, one chip of bh-lb-09 12 x 10, bfp4, LoFi, clamped_silu, capacity 8192,
E local = 48 / 24 / 12 = 2x4 / 4x4 / 4x8; the model's M at 5120 tokens = 80).
**I 2304 is unsupported**: TT_FATAL "no gate/up split for 72 tile columns" (flat_routed_expert_plan.cpp gu_split): the
per-core column count It / np (np 1..4) must be a multiple of the 16 readers - 72 / 36 / 24 / 18 are not (and np 1 is
> 64 gate/up cores). Subgrid modes (8 / 4 readers) only for It <= 32. Everything else at H 5120 runs (bracket below).
us per expert (DRAM % of 512 GB/s), balanced, slowest of the second (non-precompile) pass:
| M | I 2048: E 48 | E 24 | E 12 | I 3072: E 48 | E 24 | E 12 |
| 32 | 40.4 (89%) | 41.1 (87%) | 42.6 (84%) | 62.0 (86%) | 61.7 (86%) | 61.9 (86%) |
| 80 | 45.1 (84%) | 45.6 (83%) | 47.5 (79%) | 69.1 (80%) | 70.4 (78%) | 72.2 (76%) |
| 128 | 50.0 (79%) | 50.3 (79%) | 52.1 (76%) | 72.3 (79%) | 74.2 (77%) | 76.8 (74%) |
| 256 | 61.1 (73%) | 62.1 (72%) | 64.9 (69%) | 82.6 (75%) | 84.2 (74%) | 88.8 (70%) |
| 1024 | 169.5 (45%) | 171.8 (44%) | 176.8 (43%) | 270.7 (34%) | 273.4 (34%) | 278.1 (33%) |
Both run at ~385-390 GB/s effective at M 80, so I 2304 (19.9 MB / expert) would be ~51 us / expert with a split of the
same efficiency: ~2.45 / 1.22 / 0.62 ms a call at 48 / 24 / 12 experts. Padding to I 3072 (the K3 split) runs today
at ~70 us / expert (+37%). Natural fix: np 3 -> 24 pair-sets over the 16 readers, 3 per bank (2 + 1 per reader, banks
balanced), g 2 -> 48 gate/up cores as K3 - needs per-reader pair-set counts in the plan and readers.

### Uneven gate/up split: DeepSeek V4.1-Flash I 2304 runs (2026-10-17)
When no It / np is a multiple of the 16 readers, the plan now takes an uneven split (MIMO_FL_UNEVEN=0: off): the
pair-sets spread over the 8 banks (shares 1 apart), each bank's over its 2 readers (rgs per reader, ps0 first pair-set,
gu_off; reader kernels se_reader / se10_fwd / se9_rdown and the reader CB built per pair-set count; gate/up cores take
pair-set ps0[r] + j / G; flat_weight_regions packs per reader, zero-padded to the largest region; the weight cache
key gains rgs only for uneven plans). Receivers per reader up to 8 (credit semaphores 0..7 are free on reader cells;
5 with reader down columns: se9 uses 5 / 12). More than 64 gate/up cores: the auto-shaped rectangles (12 x 10: 74
cells), the smallest pair covering the split, extra cells idle. DRISC mode not ported yet (uneven plans use the Tensix
readers). test_flat_uneven_split.py (H 5120 I 2304 E 12 / 24, clamped SwiGLU, counts incl. full capacity, two
launches): min PCC 0.998 vs the bfp4 reference.
V4.1 shape, us per expert (DRAM %), balanced sweep, one chip, bfp4, LoFi; E 24 (E 48 / 12 within ~3%):
| plan | M 32 | M 80 | M 128 | M 256 | M 1024 |
| np 3, 24 gate/up (rgs 2 / 1) | 52.5 | 59.8 (70%) | 71.9 | 140.9 | 555.2 |
| np 2, 36 gate/up (rgs 3 / 2) | 48.1 | 51.7 (81%) | 55.1 | 70.5 | 271.6 |
| np 1, 72 gate/up, reader down cols (rgs 5 / 4), 15 down | 45.9 (88%) | 51.0 (83%) | 55.4 | 68.9 | 245.4 |
| same, all 26 down candidates (probe MIMO_FL_ND=64) | - | 51.4 (82%) | - | 67.4 | 217.1 |
Bounds: M <= 128 DRAM / latency (~80-88%, DRISC off); large M down-bound: 72 gate/up (as if 108 cores) but 15 down
cores at 8 columns (as if ~60) - the plan's trim drops down cores east of column 8 (slow for K3); keeping them gives
-12% at M 1024. Even I 2048 at M 1024: 169.5 (ideal for I 2304 at its efficiency ~190).

### Half-width gate / up tiles, step 1 (2026-10-17, MIMO_FL_HG=1)
Idea (16-column split granularity without the slow tiny-tile matmul: zz_scratch_tiny_mm.py measured 32 x 16 and
16 x 32 output tiles at the 32 x 32 tile's cycles, 17.3 / 16.1 per tile vs 17.3, i.e. half the MAC rate): a weight
tile holds 16 gate columns then the same 16 up columns, so the full-tile matmul leaves gate in faces 0 / 2 and up in
faces 1 / 3. A pair's two tiles A / B = [gate | up] of its h tile's left / right 16 columns (host: _hg_pair, a column
permutation); the activation runs face-wise (se_glu_faces.hpp calculate_glu_faces under VectorMode::C: gate at the
face, up one face on; A's result into its left half in place, B's into A's right half), so the packed h tile and
everything downstream are unchanged. SiLU / SiTU / clamped SiLU (the sigmoid is SiTU's exp + Newton reciprocal).
SFPI compiler trap: two inlined copies of the SFPU body in one loop (HALF 0 / HALF 1 calls) ICE in rvtt_synth_renumber
(also a runtime half with two call sites); one call site per unit (out tile u & ~1, half u & 1) compiles. (Repro loop:
TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1, rerun the logged compile + link from the kernel's trisc2 build dir, 0.6 s.)
test_flat_uneven_split.py with MIMO_FL_HG=1 (V4.1 E12 / E24 clamped SiLU, GLM SiLU, K3 SiTU): min PCC 0.998, the
same as the standard layout. V4.1 E24 us / expert, standard / HG: M 32 45.9 / 45.8, 80 51.0 / 51.1, 256 68.9 / 68.8,
1024 245.4 / 245.4. Next: odd 16-column units per core (V4.1: 48 cores x 3 units, 3 per reader).

### Half-width tiles step 2: 16-column units per core, odd counts (2026-10-17) -> V4.1 default
A core takes NU units ([gate | up] tiles of 16 columns each, plan nu; generic widths gwt / hpc / ngrp replace 2 np /
np / It / np in the plan, factory, se3_compute and se5_recv). Its h is a run of HPC = (NU + 1) / 2 tiles from K-tile
cg NU / 2: when cg NU is odd the first tile is only its right half (lead: the face-wise activation writes unit u to
slot((u + lead) / 2), half (u + lead) & 1; slot(j) = j ? 2 j - lead : 0), when (cg + 1) NU is odd the last tile only
its left half; se5_recv writes the full tiles as one run and the halves face by face (bfp8: 16 B exponents + 256 B
data of faces 1 / 3 or 0 / 2; bf16: 512 B faces), the down cores' gather still counts one slice per gate/up core.
Weights: _hg_units (unit u = [gate | up] of columns 16 u..), a column group's NU units in order.
Default when no even pair split exists (MIMO_FL_HG_AUTO=0: off; MIMO_FL_HG_NU forces NU): the most gate/up cores that
are a multiple of the 16 readers and fit; reader down columns off with units (they slowed the readers' streams).
DeepSeek V4.1-Flash (I 2304 = 144 units): NU 3, 48 gate/up cores (3 a reader, banks / readers exactly balanced), 40
down. test_flat_uneven_split.py (V4.1 E12 / E24, GLM, K3; MIMO_FL_HG_NU=3 also K3 with 64 x 3): min PCC 0.998.
V4.1 us / expert (DRAM %), E 24:
| plan | M 32 | M 80 | M 128 | M 256 | M 1024 |
| uneven pairs, 72 gate/up, reader tails (previous default) | 45.9 (88%) | 51.0 (83%) | 55.4 | 68.9 | 245.4 |
| units NU 3, 48 gate/up, reader tails | 46.9 | 52.3 | 55.8 | 67.1 | 242.0 |
| units NU 3, 48 gate/up, no reader tails (default) | 44.3 (91%) | 48.2 (87%) | 51.9 (85%) | 62.8 (78%) | 226.5 |
E 48 / E 12 at M 80: 47.5 (89%) / 49.8 (85%); M 1024: 225.8 / 229.7. Probes (NU 3): full-sync DST 284 (worse),
MIMO_FL_ND=64 (untrimmed down) no gain, PCD_R 3 / 4 with tails 233 / 245.

### V4.1 large M: what bounds it; bf16 h + units hangs the chip (2026-10-17)
M 1024 E 24 (units NU 3, 226.6 us / expert): probes NO_ACT 219.9, H_NODATA 210.3, X_RESIDENT 210.2, W_NOREAD +
W_NOFWD 221.8, all of them 207.5 (matmul only). The gate/up FPU is the bound: 32 x 160 x 3 tile products (matmul_block,
ct = 3, rt 1 a row pass) at ~18 cycles = ~205 us. Not SFPU overlap (-3%) nor h delivery (-7%). It is the gate/up vs down
core balance: gate/up = units x 65.6 us, down = columns x 29.5 us -> NU 3: 197 / 118; NU 2: 131 / ~177 (28 down);
NU 2 + DRISC readers (16 cells back): 131 / ~118 (~44 down). Ideal on 100 cores 142, on 116 122.

bf16 h with units (GLM_SWEEP_HBF16=1, NU 3) HANGS the chip on the first launch (bfp8 h with the same plan is fine). Twice
the whole host then rebooted: the dispatch timeout ran tt-triage (host MMIO reads into the wedged chip; plus the NOC0
retry that was in scripts/lib/tt_safe_run.sh, now removed). With no probing (no triage, no watcher, os._exit on the
timeout, no reset: scratch noprobe_run.sh) the same hang left the host up; the KMD then logged "Timeout waiting for ARC
response" on close: the hang takes ARC with it. The planner refuses NU + bf16 h (MIMO_FL_HG_BF16_UNSAFE=1: mock /
emulator only). Mock cluster (12 x 10) diff of every kernel's args (MIMO_FL_DUMP_RT=path) bf16 vs bfp8 h: same cores and
NoC targets, arena fits (564 x 2 KB from 415744); differences: h tile 2048, x slots 32 (24), y rows 2 (4) and NO
SE_DN_REG (bf16 h: dring 1.5, the unpinned down schedule, never run with the unit layout / 40 down cores). Suspects, in
order: the unpinned down schedule with units, x slots 32, the 512 B bf16 half-tile h writes.
Found (same day, one-variable A/B, probing off): bfp8 h with MIMO_FL_DRING=1.5 (the unpinned down schedule) hangs as
well, bf16 h with DRING=2 (pinned) runs, the uneven pair split (15 down, reader tails) runs unpinned, and
MIMO_FL_DW_BATCH=1 runs. Cause: se6_dw writes a batch linearly from the CB write pointer while the compute reads block b
at ring slot b % ring; a 1.5-expert ring of 27 blocks with batch 2 put every other lap's second block past the ring end
(onto the h buffers). Fix: an unpinned batch stops at the ring end (SE_DW_RING), as a pinned one stops at a load
boundary. The planner guard is gone; test_flat_uneven_split.py gained v41_e24_hbf16 and v41_e24_pin0 (min PCC 0.998).

bf16 h cost at large M (E 24 M 256 / 1024: 62.8 / 226.6 bfp8 -> 83.7 / 289.7 bf16) is all h delivery: MIMO_FL_H_NODATA
207.8; x slots 24 283.0, X_RESIDENT 278.9, LINK_DEPTH 6 297.2, HPIECES 16 298.8 (not round-trip bound). It is the
down chain layout: each gate/up core writes its slice to every chain head (7 heads), the chains forward it.
| chains (MIMO_FL_DCH) | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 12 | 14 |
| bfp8 h M 1024 | 211.7 | 211.6 | 213.4 | 232.1 | 226.6 | 230.6 | - | 236.1 | - | - |
| bf16 h M 1024 (1.5 ring) | 217.3 | 221.1 | 246.2 | 243.8 | 289.7 | 298.0 | 257.9 | 264.6 | 338.4 | 348.9 |
bf16 h + 4 chains + the 2-expert pinned ring (DRING 2): 48.8 / 63.6 / 209.1 at M 80 / 256 / 1024. Defaults for unit
plans now: 4 chains, the 2-expert ring with bf16 h too. V4.1 E 24 us / expert at M 32 / 80 / 128 / 256 / 1024:
bfp8 h 44.2 / 48.0 / 51.9 / 62.9 / 211.6 (was 226.6 at M 1024); bf16 h 44.6 / 48.8 / 52.5 / 63.6 / 208.8 (was 289.7).
M 1024 is now at the h-off floor; what is left is the gate/up FPU (NU 2 + DRISC readers, ~131 us projected).
DRISC with units (NU 3): the job descriptor is per slot (rg, R, region, receivers), nothing unit-specific; it just
runs (PCC as Tensix readers). Default now as the other plans without reader down columns: pure slots, Tensix readers
above 96 tokens / expert (MIMO_FL_HG_DRISC=0: off). E 24 us / expert M 32 / 80 / 256 / 1024: 44.2 / 48.0 / 62.9 /
211.6 -> 43.2 / 48.2 / 62.9 / 211.7 (bf16 h 43.4 / 48.2 / 63.6 / 208.8). Without the Tensix readers (MIMO_FL_DRISC_DUAL=0:
16 cells freed) 58.9 / 61.3 / 70.6 / 215.0, the same with the freed cells idle (RD_DOWN=0) or reader-distance assignment
(DRISC_ASSIGN=0): the DRISC-only weight path itself is slower here, cause not found. NU 2 (72 gate/up cores) needs the
freed cells (TT_FATAL: down arena does not fit with Tensix readers) and an uneven DRISC split (4 / 5 cores a slot);
projected M 1024 ~150 us (gate/up 131, down 32 cores x 4.6 cols) but it would pay the DRISC-only cost at small M.

### DRISC-only (all 16 slots on DRAM cores, Tensix reader cells free): what hurts it (2026-10-17, V4.1 E 24, units)
Probes in the op (the real se_drisc_gu loop), MIMO_FL_DRISC_PROF section cycles, kernel wait zones (MIMO_FL_ZONES),
NoC traces + tt-npe. us / expert; Tensix readers (the default) M 32 / 80 / 256 / 1024: 44.2 / 48.0 / 62.9 / 211.6.
- **Bug: DRISC-only never used a bank's second tile.** Switch mode's threshold (SE_DRISC_ALT_MAXCNT) was only defined
  for dual plans, so MIMO_FL_DRISC_DUAL=0 ran both of a bank's slots on one tile (profile: tile 1 of every bank 0
  units, tile 2 2880): 58.9 / 61.3 at M 32 / 80. Fixed: 43.2 / 47.5 (better than Tensix), M 256 70.6, M 1024 215.1.
- **Delivery alone is fine.** No math, x resident, no h (the call bound by weight delivery), M 80: DRISC two tiles
  44.5 vs Tensix 44.0 (both ~95% DRAM); writes only 24.8 vs 27.8 (DRISC writes are faster).
- **Mid M: tile 1's DMA delays the x reads.** M 256 DRISC two tiles 69.8: W_NOREAD 62.6, X_RESIDENT 62.6, both 55.0
  (= Tensix floor); DMA only (W_NOFWD) 68.8, writes only (W_NOREAD) 61.0; XRD_SKIP (no x DRAM reads) 59.8, XMC_NODATA
  69.3 -> the x DRAM reads vs the DRISC DMA. Relay zones (E 4 M 256, cycles per call): XRD_DRAM (read barrier) Tensix
  60k / DRISC two tiles 221k / one tile 70k; XRD_FULL 118k / 2k / 155k; TZ_IN 12k / 72k / 12k. tt-npe on the traces
  (Tensix side; the DRISC traffic is not traced): congestion impact 0%, max link 25-27% in both modes, relays issue at
  the same rate (median 113 cycles a 2 KB read) but stall 118k vs 15-38k cycles a call. The x relays read on NOC1,
  i.e. through each bank's NOC1 endpoint = DRISC tile 1: its DMA shares that tile's port with the x reads. With the
  Tensix readers the weights + down reads use the NOC0 endpoint's port and x has the NOC1 port alone.
- Things that do not fix it: MPFE weights (0,0,3 / 0,0,7 / 0,3,7 / 0,7,7 / 7,0,0 / 3,0,0: 69.8-73.2; tile 1's own port
  is the x port, so favoring x favors its DMA too; also with DMA on tile 2 only, 0,0,7 / 1,0,0: no change), DMA depth
  7 (75.5), 4 (74.5), more x reads in flight (XRD_BATCH 4: 75.7), all writes on NOC0 (71.2), x read once (X1 85.2,
  XFWD 102.6), x reads NOC0 + multicast NOC1 (DRISC_XNOC 106.5), free-on-sent slots (69.2), draining the loop.
  Partial: tile 1 throttled (ALT_BURST 4: 66.2, ALT_DMA_MAX 1: 65.3) but small M pays (M 80 ~50).
- **One tile per bank cannot carry a bank** (x is then fine, XRD_DRAM 70k): isolated delivery 59.3 at M 80
  (~28 GB/s a tile, 826 cycles a 13.8 KB block: DMA issue 186, write 199, flush + inc 158, loop 270; the tile's own
  DMA + NoC writes, 8f's 37-47 GB/s ceiling); not the loop shape (DRAIN 60.5) nor the ack latency (FREE_SENT 62.6).
- x reads on NOC0 (both relay kernels on NOC0) hang, also with Tensix readers and in dynamic-NoC mode
  (MIMO_FL_XRD_DYN=1, which alone runs): the same-NoC reads + multicast on the relay; not debugged (needs triage).
- The tensor prefetcher's author (drisc.md, the talk) uses the NOC1 endpoint tile for weights next to the free tile
  (dual senders, 60 GB/s a bank) and lists DRAM priority vs NoC traffic as open; decode has little NOC1 DRAM read
  traffic, flat's x path is exactly that.
Where it stands: DRISC-only = Tensix at M <= ~224 (two tiles), +5..12% at M 256 (two tiles: x; one tile: rate),
+1.6% at M 1024. Options: x off NOC1 (fix the same-NoC hang, or move the relays so a NOC1 multicast works), or keep a
Tensix reader for each bank's second slot (the old hybrid: 8 cells free).

### x off the DRISC tile's port: linked multicasts, helper reads on NOC0 (2026-10-17, V4.1 E 24, units)
- **The x-reads-on-NOC0 hang is the linked multicast.** The primary relay sends x as linked multicast chains on NOC0;
  with MIMO_FL_XRD_NOC=0 its BRISC issues DRAM reads from another command buffer on the same NoC while a chain is
  open, which deadlocks (the documented linked-mcast hazard; dynamic-NoC mode, MIMO_FL_XRD_DYN, does not help). With
  MIMO_FL_XMC_UNLINKED=1 (every piece unlinked) x reads on NOC0 run.
- **Unlinked multicasts cost too much**, us / expert M 80 / 256 / 1024:
  | config | M 80 | M 256 | M 1024 |
  | Tensix readers (default, linked) | 48.0 | 62.9 | 211.6 |
  | Tensix, unlinked | 48.3 | 71.2 | 258.5 |
  | Tensix, unlinked + x reads NOC0 (dynamic NoC) | 52.6 | 90.0 | 264.8 |
  | DRISC two tiles (DUAL=0, ALT_MAXCNT 4096), linked | 47.5 | 69.8 | 216.4 |
  | DRISC, unlinked | 47.9 | 78.2 | 266.1 |
  | DRISC, unlinked + x reads NOC0 | 50.4 | 80.6 | 271.8 |
  | DRISC, unlinked + x reads NOC0 + down reads NOC1 (DN_NOC0) | hang (passed standalone at M 256) |
- **Helpers read on NOC0, primaries keep NOC1 reads + linked NOC0 multicasts** (MIMO_FL_XHELP_NOC0=1: the helpers'
  reader + sender in dynamic-NoC mode; MIMO_FL_NH helpers per rectangle take (NH) / (NH + 1) of x):
  | config | M 80 | M 256 | M 1024 |
  | DRISC, helpers NOC0, NH 1 | 49.0 | 72.3 | 216.1 |
  | DRISC, helpers NOC0, NH 2 | 49.0 | 69.9 | 216.4 |
  | DRISC, helpers NOC0, NH 3 | 49.5 | 69.4 | 215.5 |
  | Tensix, helpers NOC0, NH 1 | 49.5 | 82.4 | 213.1 |
  | Tensix, helpers NOC0, NH 2 | 51.0 | 80.8 | 212.2 |
  x reads are slow on whichever bank tile also carries a bulk stream: tile 0 (the NOC0 endpoint) has the down weight
  reads (and, with Tensix readers, the gate/up weights: M 256 62.9 -> 82.4), tile 1 the DRISC DMA. Only the Tensix
  reader layout leaves x a tile to itself (tile 1, NOC1). Left: all x on NOC0 by helpers (the primaries reading
  nothing: an se11_xmc change) with the down reads on NOC1 next to the DRISC DMA, or the hybrid (free-tile DRISC +
  Tensix readers, 8 cells).
- **Helpers read all of x** (MIMO_FL_XHELP_ALL=1: the primaries only multicast, and read x themselves only on
  small-M calls; helper j takes super-blocks g % NH == j). PCC as default (V4.1 E24 cases, also with DN_NOC0).
  DRISC two tiles, us / expert M 80 / 256 / 1024 (baseline 47.5 / 69.8 / 216.4):
  | helpers read on | NH | M 80 | M 256 | M 1024 |
  | NOC1 | 1 | 47.7 | 79.4 | 238.2 |
  | NOC0 | 1 | 53.9 | 100.7 | 322.6 |
  | NOC0 | 2 | 50.2 | 78.3 | 226.6 |
  | NOC0 | 3 | 49.7 | 75.6 | 227.6 |
  | NOC0 + down reads NOC1 (DN_NOC0) | 1 | hang in the sweep (passed standalone at M 256 and the PCC test) |
  One reader core cannot carry a rectangle's x (~24 GB/s of 2 KB reads), and NOC0 x reads (tile 0, with the down
  weight reads) are slower per reader than NOC1's. DN_NOC0 hangs intermittently with these relay layouts (also with
  NH 2 standalone, and with unlinked multicasts) - the one layout that would leave x a bank tile to itself.

### V4.1 hot-expert tolerance (2026-10-17, single chip, E 24, DRISC-only, 28 down; test_flat_balanced_sweep GLM_SWEEP_COUNTS)
Patterns: one hot expert of h tokens over 23 at 32 (or 80); "mix" = h + 4 x 256 + 6 x 48 + 8 x 24 + 5 x 12; two hot;
few active. ms per call, then max(DRAM %, math % of 90 cores) (DRAM % assumes every expert active: over 100% when not).
| pattern | 48 gate/up | 72 gate/up |
| rest 32: h 32 / 1024 / 2048 / 4096 / 8192 | 1.041 / 1.117 / 1.250 / 1.490 / 2.148 | 1.237 / 1.250 / 1.268 / 1.478 / 2.122 |
| rest 80: h 80 / 1024 / 2048 / 4096 / 8192 | 1.146 / 1.255 / 1.364 / 1.670 / 2.425 | 1.265 / 1.315 / 1.372 / 1.575 / 2.184 |
| mix: no hot / h 1024 / 2048 / 4096 / 8192 | 1.138 / 1.260 / 1.365 / 1.641 / 2.325 | 1.274 / 1.408 / 1.389 / 1.588 / 2.225 |
| two hot 1024 / 4096 (+3 x 256, 19 x 32) | 1.457 / 2.431 | 1.525 / 2.276 |
| 4 x 1024 only / 1 x 8192 only | 0.931 / 1.728 (math 67%) | 0.850 / 1.558 (math 75%) |
A hot expert up to ~1024 tokens hides under the others' weight streaming (+7-10%); 2048 +20%, 4096 +45%, 8192 about
2x the no-hot time. Past ~2k the call is ~ the hot expert's own time (one 8192 expert alone: 1.73 ms, 67% of the 90-core
peak) + ~0.4 ms of the other 23 (alone ~1.0 ms: about 60% hidden under it). 72 gate/up: slower without a hot expert
(-16%), even at 2048, better from 4096 (8192: -1..-10%); a lone large expert runs at 75% vs 67%.

### Why more down cores (or gate/up cores) gained nothing (2026-10-18, single chip, DRISC-only, us / expert)
Effective-core math (min(1.5 gate/up, 3 down)) predicted +14% for K3 at 32 down and more for V4.1 at 72 gate/up.
Three ceilings sit within ~10% of each other per 64-row sub-block, so lifting one alone shows nothing:
1. **Down column granularity**: the down output is H / 32 column tiles split over the down cores - K3 112 tiles: 28
   cores x 4 exactly, 32 cores still have 4-column cores (the slowest sets the pace); time only steps at ceil(112 / ND)
   = 3 (38 cores). V4.1 160 tiles: 28 -> 6, 32 -> 5 (72 gate/up: 190.6 -> 179.6 at M 1024), 40 -> 4 (181.9: no more).
   Zones (K3, compute only, 32 down): the down cores are in SE_DOWN ~100% of the time (~14k cycles a sub-block at 4
   columns, ~88% of the LoFi tile rate), the gate/up cores 33-49% in SE_GU_MM.
2. **x delivery**: ~15.5k cycles a sub-block = 64 rows x 7 KB x 2 rectangles at ~80 GB/s, ~20 GB/s a relay (the
   known relay rate). K3 M 1024: 28 down 182.6, 38 down 186.1; x reads skipped 163.4; **MIMO_FL_NH=2 (two x helpers a
   rectangle) 165.2** (38 down 169.1, NH 3 168.4). V4.1 is not x bound (NH 2: 72/32 177.9 vs 179.6, 48/28 219.8 vs 216.5).
3. **Gate/up compute**: with x free, ~13.7k cycles a sub-block whatever the down count = the 2 x 3-tile matmul blocks
   at ~79% of the LoFi tile rate (64 gate/up cores, K 112).
Compute only (no weight / x traffic): K3 28 down 171.6, 32 down 163.7 (127 us = the LoFi floor at 96 effective cores).
In the overlap (K3 harness, accumulators on ports): NH 2 flat alone packed 5592 -> 5277, fused 5807 -> 5709 (the hot
chip's combine links then show: fused / flat 1.039 -> 1.082); uniform / spread unchanged. K3 block: NH 2 neutral
(traced 3312), now its default with acc_on_ports. Next levers: gate/up block shape (taller blocks: fewer weight unpacks
per FLOP), down cores at the next granularity step, the hot chip's combine link time.

### x delivery: unicast snakes by default (2026-10-19, TS_HANG_WORKLOG.md waves 9-14)
The relay's linked NOC0 multicast into the gate/up rectangle wedged NOC0 beside the DRISC weight writes and DRAM
traffic (a hardware hang, reproduced by a traffic-only probe). Default now (program factory XPath): each primary
unicasts its blocks into one snake head per rectangle; the gate/up cores forward them along the snake (down a column,
wrapping, one column over) with a dedicated NCRISC forwarder (se5_xfwd.cpp, NOC0, 4 cut-through pieces, 3 in
flight); the receiver BRISC (NOC1) only consumes. Multicast stays as MIMO_FL_X_CHAIN=0, and where the plan has no
full rectangles / one primary per rectangle. Perf vs the multicast (fused flat + combine, us): GLM-Flash S 256 / S
640, K3, V4.1-Flash within +1.0 .. -4.4% (V4.1 packed -4.4%); GLM-5.3 flat alone M 512 / 1024 -4.8 / -3.0%. Whole
model GLM-Flash traced chunk neutral (S 256 244.2 -> 244.4 ms, S 640 479.4 -> 479.1 ms). Bugs on the way: the
receiver's end count lagging (forwarder never ended), the relay counting a batch only on retirement (GLM-5.3's full
24-of-24 x ring deadlocked), ids polled before the command buffer drained (stale x, 1 in ~1e5 calls).

### Fused flat + combine vs the DRAM floor (2026-10-19, 2 x 4, pure DRISC, unicast x, full-L1 arena + landing pool)
Floor of the busiest chip (lean_sender/floor.py, the harness's routings): DRAM = its active experts' bfp4 weights + x
row reads (M_total x H bf16) + combine's output writes (S K rows: y reaches combine in L1) at 512 GB/s; math = LoFi
4096 FLOP / cycle / core at 1.35 GHz (assumed, not measured) on the 90 compute cores. Harness FC_DET_REPEAT=0 (the
full-L1 arena leaves no room for its sum check).
| config | routing | M_total | flat alone | fused | DRAM floor | math floor | fused / floor | DRAM % | math % 90 / 110 | us / expert |
| GLM-Flash S 640 | uniform | 5206 | 1331 | 1362 | 1161 | 579 | 1.17x | 85% | 43 / 35% | 38 |
| GLM-Flash S 640 | zipf spread | 8071 | 1607 | 1700 | 1206 | 874 | 1.41x | 71% | 51 / 42% | 47 |
| GLM-Flash S 640 | zipf packed | 29579 | 4097 | 4741 | 1551 | 3052 | 1.55x | 33% | 64 / 53% | 132 |
| GLM-Flash S 256 | uniform | 2083 | 1208 | 1229 | 1061 | 252 | 1.16x | 86% | 21 / 17% | 34 |
| GLM-Flash S 256 | zipf spread | 3216 | 1327 | 1349 | 1080 | 392 | 1.25x | 80% | 29 / 24% | 37 |
| GLM-Flash S 256 | zipf packed | 11801 | 2049 | 2215 | 1217 | 1259 | 1.76x | 55% | 57 / 47% | 62 |
| K3 (S 320, K 16) | uniform | 5246 | 2285 | 2305 | 2177 | 786 | 1.06x | 94% | 34 / 28% | 41 |
| K3 | zipf spread | 6369 | 2493 | 2515 | 2193 | 956 | 1.15x | 87% | 38 / 31% | 45 |
| K3 | zipf packed | 28854 | 5228 | 5321 | 2508 | 3955 | 1.35x | 47% | 74 / 61% | 95 |
| V4.1-Flash (S 320) | uniform | 1967 | 1093 | 1100 | 1011 | 332 | 1.09x | 92% | 30 / 25% | 46 |
| V4.1-Flash | zipf spread | 3281 | 1237 | 1248 | 1037 | 528 | 1.20x | 83% | 42 / 35% | 52 |
| V4.1-Flash | zipf packed | 10718 | 2493 | 2509 | 1186 | 1579 | 1.59x | 47% | 63 / 51% | 105 |
The combine costs 0.6-2.3% on uniform and spread (fused / flat) except GLM-Flash S 640 spread (5.8%) and packed S 640
/ S 256 (16% / 8%): the gap to the floor is the flat expert's own. GLM-Flash spread S 640: flat alone 1.33x the floor
(the hot experts' 4738 rows on one chip's grid).

### GLM-Flash uniform: why the flat expert is at ~82-86% of the floor (2026-10-19, one chip, pure DRISC 64 gate/up + 32 down)
test_flat_balanced_sweep.py (now takes layout_from_env(): the harness's MIMO_FL_CMB_DRISC=1 DRISC_DUAL=0 DRISC_PURE=1
ND=32 NH=1; the plan: 64 gate/up, 32 down, 4 relays), M 58 / 145 (S 256 / 640 uniform), E 36, us per call; floor
weights 996 + x 33 / 84 + y 33 / 84. Recipe: /localdev/mstaletovic/flat90/run.sh <tag> [probes].
| variant | M 58 | M 145 |
| base (y to DRAM) | 1208 | 1436 |
| NO_MATH | 1210 | 1430 |
| YRM_NOWRITE (no y: the fused case, floor 1029 / 1080) | 1199 | 1322 |
| + XRD_SKIP (weights only) | 1198 | 1288 |
| XRD_SKIP | 1202 | 1289 |
| W_NOREAD (weights forwarded, not read) | 1044 | 1285 |
| W_NOFWD (read, not forwarded) | 1182 | 1433 |
| read only (W_NOFWD + XRD_SKIP + YRM_NOWRITE) | 1036 | 1089 |
| H_NODATA | 1201 | 1417 |
| E 18 (base / no y) | 621 / 606 | 744 / 693 |
| no y + DRISC_DMA_MAX 4 / 8 | 1199 / 1200 | 1356 / 1367 |
| no y + DRISC_UB 2 / 4 (blocks per DMA unit) | 1122 / 1158 | 1394 / 1385 |
| no y + DRISC_ALT_DMA_MAX 1 / ALT_BURST 4 / XHELP_NOC0 | 1389 / 1288 / 1199 | 1485 / 1418 / 1320 |
| X_RESIDENT | hang (triage: flat90/xres_triage.txt) | |
- Not compute (NO_MATH unchanged). No y: 14 / 65 us fixed + 32.9 / 34.9 us an expert (floor 28.6 / 30.0): ~87% of 512 GB/s
  in steady state at both M, plus a pipeline tail at M 145.
- The DRAM reads alone run at 96% (read only 1036); forwarding alone near the floor too (W_NOREAD 1044); both on one
  tile: 1198 = 16 tiles x ~26.6 GB/s - the per-tile DMA-in + NoC-write-out loop (~600 of ~830 cycles a 13.8 KB block
  fixed), the "one tile ~28 GB/s" ceiling. Single-block units (pure DRISC: one job per tile) pay it per block: UB 2
  takes M 58 to 1122 (92% of the fused floor), but M 145 to 1394 (+5%: half the slots in flight, NS = room / (UB blk)).
- M 145 also: x reads cost 147 us against 84 of floor (the x relays' NOC1 reads vs each bank's NOC1-endpoint DRISC
  tile), and the 65 us tail (the last experts' gate/up -> down after their weights).
Options: UB chosen per call from the rows (runtime, like the ALT_MAXCNT switch), or UB 2 with the slot room kept; the
bank's third tile as a forwarder (if free); the tail / x port at M 145.

### x injectors on the gate/up cores instead of relays: microbenchmarks (2026-10-19, one chip, nothing else running)
zz_scratch_x_injector.py (flat90/xi.sh): a piece = 32 random x rows (T 5120, H 4096 bf16 row-major in DRAM) x W tiles
of columns; tilize = fast_tilize bf16 -> bfp8 (se11_tz.cpp's); inject = gather + tilize + write the bfp8 tiles to one
other core's L1. GB/s of bf16 x.
| case | result |
| tilize, W 8 / 16 / 32 | 33.7 / 32.9 / 32.4 cycles a tile (~83 GB/s a core) |
| gather, 1 core, 512 B a row (W 8) | 11.2 (1 piece in flight) / 13.5 (4): ~51 cycles a read, issue bound; NOC0 = NOC1 |
| gather, 1 core, 1 / 2 KB a row (W 16 / 32) | 26.9 / 52.6 |
| gather 2 KB rows, 8 / 16 cores, NOC0 | 335 / 368 (row placement), 342 / 353 (spread over rows) |
| gather 2 KB rows, 8 / 16 cores, NOC1 | 95 / 155 (side by side on one / two rows), 223 / 228 (spread) |
| inject 16 cores -> one core, NOC0 reads | 155 / 149: the one receiving core's ingress (~80 GB/s of bfp8) |
- Tilizing on the gate/up cores is free: M 145 = 73 tiles an expert for an injector doing 1/8 of x (~1.8 us of a 35 us
  expert, math ~40% busy), ~0.2 us with all 64.
- Reads must be >= 1-2 KB a row (the issue cost); NOC1 DRAM reads cap at ~225 GB/s spread and less for cores sharing
  rows, NOC0 ~350; one receiving core takes ~150 GB/s of x (so >= 2 snake heads for a hot expert's ~85-100 GB/s).
Not yet measured beside the DRISC weight stream (NOC1 = the DRISC tile 1 port; the full op's NOC0 x reads hang).
- The full op's MIMO_FL_XRD_NOC=0 hang was the relay's reader and sender both on NOC0 (dedicated-NoC mode: one RISC
  per NoC). Now reads on NOC0 move the sender to NOC1 (unicast chain only; TT_FATAL on a shared NoC). It runs, no gain:
  no y M 58 / 145 1199 / 1331 (NOC1 reads 1199 / 1322); with y 1221 / 1554 (1208 / 1436: the y writes vs the NOC0
  reads). In the fused case (no y) x reads cost only ~34 us at M 145 (XRD_SKIP 1288 vs 1322) and ~0 at M 58 - the
  147 us earlier was with the y writes. x is not what keeps uniform GLM-Flash under 90%.
- Unit size, profiled (MIMO_FL_DRISC_PROF + flat_drisc_dump after each M, GLM_SWEEP_DRISC_DUMP=1; no y). Pure DRISC
  without dual has no switch (ALT_MAXCNT 0xFFFFFFFF: both of a bank's tiles every call). The profile's timestamp reads
  alone slow UB 1 from 1199 / 1322 to 1821 / 1865 us (M 58 / 145) and UB 2 not (1122 / 1360): single-block units are
  bound by the per-unit cost on the DRAM core. Per tile, cycles of the last call (units 2304 UB 1, 1152 UB 2):
  | | dma | refresh (grant waits) | write | flush + inc | total |
  | UB 1 M 58, tile 1 / free tile | 468k / 583k | 250k / 30k | 455k / 460k | 345k / 454k | 2.19M / 2.41M |
  | UB 2 M 58 | ~300-335k | 21-78k | ~290-390k | ~260-280k | 1.38-1.47M |
  | UB 2 M 145 | ~275-330k | 205-410k | ~290-450k | ~235-285k | 1.70-1.75M |
  UB 2 unprofiled M 58: 657 cycles a 13.8 KB block (UB 1: 702), ~29 GB/s a tile: mostly per byte, not per unit -
  16 tiles cap near ~460 GB/s (~90%). UB 2 at M 145: the DRISCs wait 12-24% of the call on the gate/up cores' grants
  (2-5% at M 58): the receive side limits there. Next: what the gate/up cores wait on at M 145 (zones).
- Why UB 2 loses at M 145: device-profiler zones (MIMO_FL_ZONES=1 MIMO_FL_WAITZ=0, flat90/zones.py; W_X / W_W
  sampled, compare between runs), last call, no y:
  | M 145 | UB 1 | UB 2 |
  | gate/up waits on x (W_X) / weights (W_W) | 36.5 / 8.2 us | 56.3 / 3.4 |
  | relay DRAM read barrier (XRD_DRAM) / tilizer starved (TZ_IN) | 142 / 77 | 166 / 103 |
  | relay sender waits on the tilizer (XMC_SB) / on credits (XMC_CRED) | 39 / 27 | 219 / 2 |
  With paired DMA units x becomes the bottleneck: the relays' gathers slow, the tilizer starves, the sender waits, the
  gate/up cores wait on x and hold their weight slots (the DRISCs' grant waits). Not the NOC1 port: UB 2 with the relay
  reads on NOC0 1131 / 1435 (NOC1 1122 / 1394); XRD_BATCH 4 1120 / 1439; DRISC_BURST 64 1122 / 1387 (XRD_BATCH 8:
  TT_FATAL, must divide 4). Likely the DRAM side: the weight stream near the practical peak (read only ~492 GB/s)
  leaves the latency-bound x gathers (7% of the bytes) queued. Next: x priority at the DRAM (MPFE), or x read further
  ahead (relay buffering), with UB 2.
- x priority / read-ahead with UB 2 (one chip, no y, M 58 / 145 us; UB 2 default weights 1122 / 1394): x reads on
  NOC0 with their port first (MPFE 1,7,1 / 3,7,3) 1138 / 1389, 1136 / 1399; relay RM_CHUNKS 8 1120 / 1383; gate/up x
  ring 32 slots 1121 / 1390 (SB_SLOTS 6: relay arena does not fit) - neither helps. The GDDR port weights (DRISC
  port, NOC0 endpoint, NOC1 endpoint) do: 1,1,7 1127 / 1305, 1,1,5 1119 / 1301 (best), 1,1,3 1116 / 1310, 2,1,7
  1149 / 1338, 1,0,7 1179 / 1386, 7,1,1 1129 / 1429; UB 1 + 1,1,7 1222 / 1310 - the free tile's own port held back
  vs the NOC1 endpoint (tile 1's DMA and the x reads together). Fused harness (8 chips): UB 2 + 1,1,5 S 256 uniform /
  spread fused 1229 -> 1192, 1349 -> 1303 (-3%), S 640 1362 -> 1500, 1700 -> 1837 (+10%: the combine hidden 98 -> 85%).
  S 640 uniform: UB 2 alone fused 1531 (flat 1403), 1,1,5 alone 1391 (flat 1319), UB 2 + 1,0,5 1564, + 1,5,5 1543 -
  UB 2 is what hurts the fused S 640 (not the weights). Net: no default change; per-call UB (rows <= ~100: UB 2) would
  take S 256 -3% and leave S 640.

### Per-expert weight layout is independent of experts-per-chip (2026-10-21)
For a per-expert weight cache (moe_knowledge_dump.md section 14): `zz_scratch_flat_epc_layout.py` plans each model's
shape on bh-lb-09 (12 x 10 grid, pure DRISC, MIMO_FL_RDOWN=0: no Tensix readers) at two experts-per-chip values and
lays out the same two random experts with `flat_weight_regions`. K3 28 / 56, K2.7 12 / 24, GLM-5.3 8 / 16, M3 4 / 8,
GLM-5.3-Flash 9 / 18: every layout field equal (np, rg, nu, nk_gu, nd, pcds, col0s, banks, dw_rot, ...), the experts'
blocks bit-identical, only the down regions' zero padding grows with the count (K3 1344 -> 2688 tiles per down
region). A chip's tensor = per-expert blocks placed into its regions; nothing per mesh.

## GLM-5.3 flat expert vs the DRAM floor: the DRISC per-tile ceiling again (2026-10-21, one chip bh-lb-09 12 x 10)
test_flat_balanced_sweep.py, E 32, H 6144, I 2048, SiLU, bfp4, pure DRISC 64 gate/up + 32 down + 4 relays
(MIMO_FL_CMB_DRISC=1 DRISC_DUAL=0 DRISC_PURE=1 ND=32 RDOWN=0), balanced M per expert; recipe
/localdev/mstaletovic/flat90_glm53/run.sh <tag> [probes]. Floors at 512 GB/s: weights 32 x 21.2 MB = 680 MB 1327 us,
x 61 / 123 us, y 61 / 123 us (M 80 / 160) -> 1450 / 1573 us (1389 / 1450 without y).
| variant | M 80 | M 160 |
| base | 1668 (87%) | 1919 (82%) |
| NO_MATH | 1683 | 1938 |
| no y (YRM_NOWRITE: the fused case) | 1617 | 1785 |
| + W_NOREAD (weights forwarded, not read) | 1404 | 1727 |
| + W_NOFWD (read, not forwarded) | 1528 | 1706 |
| read only (W_NOFWD + XRD_SKIP, no y) | 1385 (491 GB/s) | 1439 |
| no y + DRISC_UB 2 | 1561 (-3.5%) | 1868 (+4.6%) |
| no y, E 16 | 822 | 939 |
- Not compute (NO_MATH unchanged). No y, steady state from E 16 / 32: 49.7 / 52.9 us an expert against a floor of
  43.4 / 45.3 (87 / 86%), fixed 27 / 93 us.
- M 80: reading alone 1385 (96% of 512) and forwarding alone 1404 are at the floor; both on one tile 1617 = 3072
  units of one 13.8 KB block per tile per call in 0.53 us each = 26.2 GB/s a tile x 16 tiles = 420 GB/s: the
  per-tile ceiling of the GLM-Flash 2026-10-19 entry. DRISC profile (MIMO_FL_DRISC_PROF, the profiled call 2437 us),
  cycles per unit, the bank's two tiles: DMA 214 / 254, grant refresh 94-108 / 11-16 (470-650 vs 84-108 refreshes a
  call), NoC writes 199 / 201, flush + semaphore 155 / 197, total 984 / 1048 (a third of it outside those sections).
  One RISC does all of it per block, serially.
- DRISC_UB 2 halves the per-unit cost but halves the units in flight: -3.5% at M 80, +4.6% at M 160 (as GLM-Flash).
- x is not it (no y in all): XRD_SKIP (no x DRAM reads) 1608 / 1740, XRD_SKIP + XMC_NODATA (no x data sent either)
  1608 / 1737 vs 1617 / 1785: x costs 9 / 48 us. With x gone the weights alone run at 423 GB/s at M 80 (the ceiling)
  and 391 at M 160; the extra ~130 us at M 160 is mostly the pipeline tail (93 us fixed from E 16 / 32) plus a slower
  steady state (52.9 vs 49.7 us an expert). (The read-only vs no-forwarding gap at M 160, 1439 vs 1706, is not x.)
Options: fewer RISC cycles a block on the DRISC (no ack wait per unit, batched flushes, a multicast write where a block
goes to G > 1 receivers, fewer grant refreshes on tile 1); UB from the call's rows (2 at small M, 1 above); the tail at
large M.

### What limits the DRISC tile (GLM-5.3, same recipe; x off = XRD_SKIP + XMC_NODATA, no y = YRM_NOWRITE)
Corrections to the entry above: GLM-5.3's DRISC blocks are 9216 B (R 4 receivers, G 1), and the DRISC tiles carry
only gate/up (453 MB a call; the down cores read their own 226 MB): 3072 units of one block per tile per call, 28.3 MB
a tile -> 17.6 GB/s a tile in the real path (16 tiles), 20.2 when it only forwards.
| variant (us, M 80 / 160) | x on, no y | x off, no y |
| UB 1 (default) | 1617 / 1785 | 1608 / 1737 |
| DRISC loop only (W_NOREAD + W_NOFWD: no DMA, no writes, the grant / flush / semaphore protocol) | - | 840 / 1419 |
| units in flight 7 -> 9 (new probe MIMO_FL_DRISC_NS_MAX=14: one job a tile takes transaction ids 1-14; the L1 room, 91 KB, holds 9) | - | 1608 / 1737 |
| UB 2 (4 units of 18 KB in flight) | 1561 / 1868 | 1412 / 1747 |
| read only (no forwarding) | - | 1385 / 1420 |
- Not bandwidth (each half alone at the floor), not units in flight (7 -> 9 changes nothing), not the slow write path
  (BH NOC_MAX_BURST_SIZE 16 KB: a 9 KB block is one fast_write), not the loop's bookkeeping alone (840 us).
- Per unit: UB 2 halves the DMAs, the flush checks and the passes (the writes and semaphore increments stay one per
  receiver) and takes M 80 x off from 1608 to 1412, the read-only time. So the tile's limit is its RISC's per-unit
  work: the DRISC profile's ~1000 cycles a unit are mostly register (MMIO) accesses to its DMA engine and NIU - issuing
  the DMA and polling its outstanding count (DMA section ~230), the write command (~200), polling the transaction id's
  ack counter + the semaphore increment (~175).
- With x on, UB 2 gains only 1617 -> 1561: the next limit is x (the relays' NOC1 DRAM reads against the bank's
  NOC1-endpoint DRISC tile, drisc-endpoint-tile memory), which UB 1 hid. M 160 is bound elsewhere: no weights moved at
  all (loop only) is already 1419.
Next: UB 2 where the rows are few (per call, like the ALT_MAXCNT switch); fewer MMIO accesses a unit (outstanding-count
polls only when the slots are full, one flush poll for several units, one semaphore increment of n); then the x / NOC1
conflict, which caps what UB 2 can show with x on.

### Fewer register accesses a DRISC unit (probe MIMO_FL_DRISC_LEAN, GLM-5.3, same recipe)
Level 1: the DMA's source-high register written once a call (bank-local sources: always 0) and the attribute word
built once a job; the DMA completion count re-read only when the known completions are used up (the stream completes
in order). Level 2 (measure only, unsafe in general): no ready poll before the semaphore increment. y at M 128
bit-exact against the default at both levels.
| us, M 80 / 160 | default | LEAN 1 | LEAN 2 |
| real path (x, y) | 1667 / 1926 (M 128: 1821) | 1685 / 1929 (1817) | 1685 / 1927 (1817) |
| x off, no y | 1608 / 1737 | 1551 / 1734 (-3.5%) | 1536 / 1734 (-4.5%) |
| x off, no y, UB 2 | 1412 / 1747 | 1412 / 1746 | - |
- Where the DRISC binds (x off, M 80) one write and ~one read fewer a unit is -3.5%, one more read -1%: the reads
  (round trips) cost, the posted writes little. M 160 is bound elsewhere; with UB 2 the DRISC is no longer the limit.
- On the real path it buys nothing: as with UB 2 (1617 -> 1561 with x vs 1608 -> 1412 without), once the DRISC gets
  faster the x traffic against the DRISC tiles takes over. That conflict is now what gates the real path.
Kept as a probe (not the default: no gain where it matters; level 2 is not safe).

### x vs the weight stream on GLM-5.3's real path: the known knobs (2026-10-21)
Real path (x, y), M 80 / 160 us: default 1667 / 1926; UB 2 1672 / 2116; GDDR port weights 1,1,5 (MIMO_FL_DRISC_MPFE,
GLM-Flash's best at M 145 with UB 2) 1703 / 2007; UB 2 + 1,1,5 1746 / 2060; + LEAN 1 1751 / 2057. All worse. With
the earlier no-gain list (x reads on NOC0: hang or no help; MPFE alone; relay RM_CHUNKS / x ring depth / XRD_BATCH;
DMA caps; free-on-sent; drain) the knob space is exhausted: the x gathers (row reads into a saturated GDDR, ~7% of
the bytes) queue behind the bulk streams whatever the NoC, port weight or depth. Gap to the floor: 217 / 353 us
(1450 / 1573 with x and y at 512 GB/s). What is left is structural: x not read from DRAM in the experts window (the
dispatch lands rows in L1 in the experts' order), or x read as bulk (expert-contiguous, pre-tilized bfp8 from the
dispatch; possibly by the DRISC in the same queue as the weights) instead of row gathers.

### x through the DRISC's queue (option A), emulated: orderings, queues, the lean registers (2026-10-21, GLM-5.3)
Probe MIMO_FL_DRISC_XEMU=N[:B] + MIMO_FL_DRISC_XEMU_OPT="mode,k,cap,other,first,w": each DRISC tile also DMAs x's
share of reads from its own bank (12 KB, 2 M a tile per call = x's volume; re-reads of other weight blocks of the
bank: same DRAM load) into a scratch slot; with w each completed read is also written out (stock NoC write on the read
command buffer, transaction id 15) to a receiver's landing ring = the DRISC -> relay delivery. Relays' x reads off
(XRD_SKIP) in every row; real path otherwise (y on); garbage x, so perf only. Accounting check (real x + the extra
reads on the job's stream, UB 2, LEAN 1, M 128): y bit-exact vs the default.
| us, M 80 / 160 | |
| default (relays read x over NOC1) | 1667 / 1926 |
| UB 2 + LEAN 1, no x reads at all | 1552 / 1750 |
| ... x spread / at each load's start / one load ahead | 1552 / 1752, 1552 / 1750, 1552 / 1748 |
| ... 4 in flight / x before the weights / on the other stream (triage off) | 1555 / 1750, 1553 / 1749, 1553 / 1748 |
| ... + each x row written to a receiver: at the load / ahead / spread / cap 4 / without LEAN | 1553 / 1751, 1553 / 1750, 1555 / 1750, 1554 / -, 1553 / 1750 |
| UB 1 + LEAN 1, x spread / at the load | 1649 / 1790, 1669 / 1790 |
| UB 1 + LEAN 1 + writes, at the load | HANG at M 80 (triage off, host up; not narrowed - UB 1 is not the configuration) |
- **WRONG, see the correction below:** every UB 2 row above issued no x reads at all (the scratch slot did not fit
  past UB 2's 5 x 18 KB slots, room 93696 B, and the probe switched itself off silently) - they measure XRD_SKIP only.
  Only the UB 1 rows issued x reads.

### Correction: option A emulated with every x read issued (2026-10-21, GLM-5.3, real path, y on, relays' x off)
The probe now takes one weight slot when the scratch does not fit (UB 2: 5 -> 4 units in flight) and reports its state
in the DRISC dump (ns, scratch slot, room, x reads issued); the no-x baselines are held to the same slots
(MIMO_FL_DRISC_NS_MAX now also lowers the cap). Issued counts checked on every run (1760 / 3520 over 11 calls = x's
volume; spread rounds the interval: 1870 / 4224).
| us | M 80 | M 160 |
| default (relays read x over NOC1) | 1667 | 1926 |
| no x reads: UB 2 4 slots / UB 2 + LEAN / UB 1 | 1556 / 1552 / 1613 | 1752 / - / 1740 |
| UB 2 + x: spread / at each load / one load ahead | 1603 / 1586 / 1586 | 1841 / 1821 / 1827 |
| UB 2 + x at each load: cap 8 / x first / LEAN / other stream | 1582 / 1583 / 1590 / 1575 | 1823 / 1833 / 1822 / 1813 |
| UB 2 + LEAN + x as 2 KB segments (6 a row, cap 8: the relays' real granularity) | 1577 | 1816 |
| UB 1 + LEAN + x at each load | 1671 | 1790 |
| + each x read written out by the DRISC (to a receiver's landing ring) | HANG (UB 1 and UB 2) | - |
- x through the DRISC queue is not free: +20..47 us over no x at M 80, +38..89 at M 160. It still beats the default:
  best M 80 1575 (-5.5%, UB 2, other stream), M 160 1790 (-7.1%, UB 1 + LEAN). At each load (or ahead) beats spread;
  depth, x first and LEAN change little; 2 KB segments cost the same as 12 KB rows (the per-unit work does not bind at
  UB 2 + 4 slots).
- The writes are not measured: the write emulation hangs (garbage into the receivers' landing rings - not the real
  target; not narrowed). So A's projected gain is ~90 us (M 80) / ~136 us (M 160) less the DRISC -> relay writes.

### Option A for real: x from the DRISCs (MIMO_FL_DRISC_X, 2026-10-21, GLM-5.3, one chip, M 128)
Implemented end to end (probe, off by default): the launcher writes each slot's x info (the call's schedule entries,
the relays, the addresses) into its DRISC's L1 next to the job descriptor; each DRISC walks the relays' chunk stream
(entries, sub-blocks, super-blocks, row tiles) and for the rows of its bank (bank = token % 8, half = token / 8 % 2)
DMAs the 2 KB segment on its job's stream (the emulation's in-order accounting), writes it into the relay's row-major
chunk once the relay's credit covers it, then (one flush check over every x write, transaction id 15, after the
command buffer drained) bumps the relay's arrival count for that chunk slot. The relays read no DRAM: they publish a
credit per reserved batch and wait for each slot's rows. Credits and arrivals are program semaphores on the relays
(ids 0 / 1, 2, 6, 11, 12: zero at each launch, so nothing carries across calls - the op's arena is per launch and
garbage, which hung the first, arena-word version).
- **Correct:** y bit-exact vs the default path at M 128, also with x read once (X1). Default path unchanged and
  bit-exact after the change (1667 / 1824 / 1928 us at M 80 / 128 / 160).
- **Slow:** M 128 default 1821 us; DRISC x 4249 (credit re-read every 8th pass), 3348 (every pass), 3883 (XRD_BATCH 1),
  X1 3074 (vs X1 default 1801: half the writes and increments save only 274 us).
- **Why (DRISC counters, per call per tile):** ~1500 x tasks, ~1600 loop passes, ~2900 credit stalls, ~1500 credit
  re-reads (each a blocking NoC round trip). Every 32-row chunk needs rows from all 16 DRISCs while a relay holds 4
  chunks: DRISCs and relays move in lockstep, one round trip per chunk, and the stalls hold the weight stream.
- What it would take: far deeper relay rings (L1, more than 5 semaphores), non-blocking credit reads, out-of-order
  writes across relays - and even then each 2 KB segment costs the DRISC ~25-40 register accesses (DMA, write,
  increment), against a read-side gain of ~90 / 136 us (M 80 / 160) in the emulation. Not pursued further: the DRISC is
  the wrong engine for x's scatter-gather; kept as a probe.

### fp8 x (half the bytes), emulated (2026-10-21, GLM-5.3, one chip, real path)
Probe MIMO_FL_XRD_HALF: the relays read the first half of each 2 KB segment - the same requests at fp8 x's bytes
(garbage x, perf only). us, M 80 / 160: default (bf16 x) 1667 / 1926; half bytes 1615 / 1800 (-3.1 / -6.5%); no x
reads at all 1612 / 1742; half + UB 2 1574 / 1851. So x's cost is bytes as well as queueing: half the bytes recovers
95% of it at M 80 and 68% at M 160; with UB 2 at small M another 2.5% (UB 2 still loses at M 160). In the block it also
halves the dispatch's fabric bytes and its x write to DRAM. Open: fp8 x needs the dispatch to send fp8 rows and the
relays' tilizer to read fp8 (to the bfp8 tiles the gate/up cores take) - the accuracy of fp8 then bfp8 x needs the
golden check.

### Close-out defaults: DRISC only and even splits (2026-10-22, bh-lb-09 12 x 10; EXPERTS_COMBINE_CLOSEOUT.md)
The plan's defaults are now the close-out layout (FlatLayoutOptions): DRISC only (drisc_dual false, drisc_pure 1,
drisc_in_overlap true: every gate/up weight from the DRAM cores, no Tensix readers, no per-call fallback, inside the
combine overlaps too) and `even_split` (new, default true):
- **Gate/up** over the 2 I weight tiles a K row ([gate | up] 16-column units) on at most 64 cores: the fewest tiles a
  core that divide evenly over a multiple of the 16 weight slots; whole gate / up tile pairs (the older NP split) where
  they give the same tiles a core, 16-column units otherwise. GLM-5.3 / -Flash / K2.7 64 x 2 (pairs, NP 1), K3 / M3
  (I 3072) 64 x 3 units (the old first-fit picked NP 2: 48 cores x 4 tiles), V4.1-Flash (I 2304) 48 x 3 units.
- **Down** over H on at most 32 cores, the fewest cores at the fewest columns a core: GLM-5.3 / M3 32 x 6, K2.7 32 x 7,
  GLM-5.3-Flash 32 x 4, V4.1-Flash 32 x 5, K3 (Hr 3584) 28 x 4. No reader down columns (K2.7 / GLM-5.3 used them).
- Preset `LEGACY` (flat_layouts.py) restores the old defaults; the research builder's plan-equality test uses it.
- The 16-column units' face-wise GLU gained SwiGLU-OAI (MiniMax M3: (clamp(u, 7) + 1) g sigmoid(1.702 g), g =
  min(gate, 7)); the factory had refused units for it.
- `role_bytes` in the plan dict: the arena each role uses (gate/up, down, relay, reader), for the free-L1 report.

Checked (correctness only, no perf): tests/unit/test_closeout_flat.py - GLM-5.3, GLM-5.3-Flash, K3, K2.7, M3 at the
Galaxy 8x4 per-chip experts (8 / 9 / 28 / 12 / 4) and uniform counts for chunks 5120 / 16384 (M 91 - 512), two
launches: every expert's PCC >= 0.998 against the bfp4 reference. V4.1-Flash is planned only (test_closeout_plans.py).
Existing unit tests with the new defaults: uneven split (V4.1 E12 / E24 / bf16 h / pin 0, glm / k3 E12), y row-major
(mimo, k2) and indexed shapes / determinism (mimo, k2) pass. Not runnable on bh-lb-09's 12 x 10 grid whatever the
defaults: test_flat_routed_expert_op (the Python builder asserts an 11 x 10 grid) and the tp4 / tp2 cases (2-3
subgrids, which the overlap-era factory no longer builds).

### DRISC sender: posted writes, the tensor prefetcher's scheme (2026-10-22, bh-lb-09 12 x 10, flat alone, pure DRISC)
Idea from John Bauman's tensor prefetcher (PR 59934, tensor_prefetcher.cpp): posted NoC writes, one flush per slot
reuse instead of an ack check per unit. New probe levels of MIMO_FL_DRISC_LEAN (kernel only, se_drisc_gu.cpp):
- **3** (on top of 1): the block writes POSTED (no transaction id, no acks); the receivers' semaphore increments on the
  same static VC as the data (FWD_VC), issued once the NIU's posted-write-sent counter has passed the unit's last write
  (same VC and route: they arrive behind the data - the noc_async_writes_flushed + semaphore_inc pattern). One counter
  read a pass completes every unit that left, in order, and frees its slot. Without transaction ids the units in
  flight are bound only by L1: MIMO_FL_DRISC_NS_MAX up to 16 (10 fit at one 9 KB block a unit). Safe; y bit-exact.
- **4**: 3 without the ready poll before an increment - HANGS (the increments go out back to back, so the command
  buffer is overwritten while busy). Measure-only, dropped.

us per call (RT profiler, 10 iters; runs repeat within 0.1%), y bit-exact vs the default in every row:

| GLM-5.3 E 32, M 80 / 128 / 160 | real x path | x off (no x reads / x data / y writes) | half-bytes x (fp8 x) |
|---|---|---|---|
| default | 1665 / 1822 / 1923 | 1608 / - / 1734 | 1615 / - / 1798 |
| LEAN 1 | 1683 / 1816 / 1930 | 1551 / - / 1734 | - |
| LEAN 3 | 1713 / 1813 / 1930 | 1482 / - / 1681 | - |
| LEAN 3, NS 10 | 1682 / 1811 / 1927 | **1461** / - / **1677** (-9.1 / -3.3%) | 1590 / - / 1796 (-1.5 / -0.1%) |
| UB 2 | 1680 / 1866 / 2116 | 1412 / - / 1744 | 1576 / - / 1848 |
| LEAN 3 + UB 2 | 1678 / 1879 / 2094 | 1423 / - / 1680 | - |

GLM-5.3-Flash (E 9, M 64 / 128): default 329 / 388, LEAN 1 328 / 385, LEAN 3 329 / 386, LEAN 3 NS 10 328 / 384,
UB 2 335 / 406, LEAN 3 + UB 2 336 / 392. K3 (Hr 3584, I 3072, E 28, M 91 / 128): default 1230 / 1306, LEAN 1 1230 /
1315, LEAN 3 1227 / 1307, LEAN 3 NS 10 1230 / 1308, UB 2 1299 / 1414, LEAN 3 + UB 2 1300 / 1408.

- Where the DRISC binds (x off) posted writes are the biggest lever found for it: -8 / -9% at M 80 and -3% at M 160,
  and they remove UB 2's loss at M 160 (1744 -> 1680). The per-unit ack check, not the DMA, was the DRISC's cost.
- On the real x path nothing moves by 1%: x binds (GLM-5.3 M 80 +1.0%, M 128 / 160 -0.5 / -0.2% with NS 10;
  GLM-Flash and K3 within 0.3%). UB 2 loses everywhere on the real path (K3 +6%).
- Kept as a probe (LEAN 3, NS 10). Revisit with fp8 x: with half the x bytes it is already -1.5% at M 80.

### Utilization sweep, six models at Galaxy 8x4 experts per chip (2026-10-22, bh-lb-09 12 x 10, flat alone)

Current defaults (pure DRISC, even splits, default kernel, real x path), balanced M (every expert the same rows),
`models/demos/glm53_flash_d_p_lb/tests/test_flat_balanced_sweep.py` (RT-profiler device time, median of 10 calls,
1.35 GHz). Experts per chip: GLM-5.3 8, GLM-5.3-Flash 9, K3 28, K2.7 12, M3 4, DSv4.1-Flash 12. DRAM % = (bfp4
weights + x bf16 read once + y bf16) / 512 GB/s; math % against LoFi peak on 90 compute cores / the 110-core grid (the
plans' real compute cores: 96, K3 92, DSv4.1 80). Scripts and logs: /localdev/mstaletovic/flat_util_sweep/.

Plans: gate/up 64 x pairs (GLM-5.3, GLM-Flash, K2.7), 64 x 3 16-column units (K3, M3), 48 x 3 units (DSv4.1); down 32
(K3 28); 4 relays, 1 Tensix launcher.

us per call / DRAM % / math % (90):

| M | GLM-5.3 | GLM-5.3-Flash | K3 | K2.7 | M3 | DSv4.1-Flash |
|---|---|---|---|---|---|---|
| 32 | 408 / 84 / 10 | 312 / 83 / 9 | 1110 / 94 / 11 | 695 / 87 / 10 | 297 / 86 / 10 | 530 / 91 / 10 |
| 64 | 436 / 82 / 18 | 330 / 81 / 18 | 1169 / 91 / 20 | 735 / 85 / 18 | 326 / 80 / 18 | 570 / 87 / 19 |
| 96 | 468 / 79 / 25 | 356 / 78 / 25 | 1233 / 89 / 29 | 784 / 82 / 26 | 354 / 75 / 25 | 609 / 84 / 27 |
| 128 | 516 / 74 / 30 | 389 / 74 / 30 | 1308 / 85 / 36 | 829 / 80 / 33 | 375 / 73 / 31 | 656 / 80 / 33 |
| 160 | 539 / 73 / 36 | 410 / 72 / 36 | 1434 / 80 / 41 | 875 / 79 / 39 | 390 / 72 / 37 | 711 / 76 / 38 |
| 192 | 572 / 71 / 41 | 435 / 70 / 40 | 1503 / 78 / 47 | 927 / 77 / 44 | 411 / 69 / 43 | 761 / 73 / 43 |
| 256 | 642 / 67 / 48 | 494 / 65 / 47 | 1692 / 72 / 56 | 1062 / 71 / 51 | 457 / 65 / 51 | 880 / 67 / 50 |
| 384 | 787 / 61 / 59 | 598 / 60 / 58 | 2244 / 59 / 64 | 1350 / 62 / 60 | 555 / 58 / 63 | 1088 / 60 / 60 |
| 512 | 960 / 55 / 65 | 718 / 55 / 65 | 2791 / 51 / 68 | 1670 / 55 / 65 | 661 / 53 / 70 | 1344 / 53 / 65 |
| 1024 | 1722 / 42 / 72 | 1283 / 42 / 73 | 4999 / 36 / 76 | 3138 / 40 / 69 | 1159 / 38 / 80 | 2589 / 37 / 68 |
| 2048 | 3300 / 34 / 75 | 2462 / 34 / 76 | 9849 / 27 / 77 | 6077 / 32 / 72 | 2131 / 30 / 87 | 5068 / 29 / 69 |

- DRAM-bound up to M ~256-384, math-bound from ~384 (the crossover is at 384 for every model).
- A fixed ~40-95 us per call above the weight floor even at M 32 (GLM-5.3 64, GLM-Flash 54, K3 69, K2.7 93, M3 42,
  DSv4.1 48 us): pipeline fill / drain. It is 10-16% of the call for the few-expert plans (GLM-5.3 8, GLM-Flash 9,
  M3 4 experts) and 6% for K3 (28), which is why K3 reaches 94% and the others 83-86% at M 32.
- The Galaxy 5120 range (M 80-160): GLM-5.3 / GLM-Flash / M3 at 72-80% DRAM with math at 25-37%: neither bound is
  near full; that middle range is the weakest. GLM-5.3 at 8 experts and M 160 is 73% (539 us) against 82% at 32
  experts (the earlier sweeps): the fixed cost and x weigh more with fewer experts.
- Math tops out at 69-77% of the 90-core peak (M3 87%); DSv4.1's 69% is on 80 real compute cores (78% of those).
- DSv4.1-Flash (48 x 3 16-column units) shows no trace of the old ~33% DRISC-only slowdown at M 32-80 (91 / 87% DRAM,
  above the pair-split GLM models): no LEAN 3 run needed.
- Correctness: test_closeout_flat (GLM-5.3, GLM-Flash, K3, M3 at chunks 5120 / 16384, K2.7 at 5120) plus DSv4.1-Flash
  at both chunks (the same check, run ad hoc): 11 passed, min PCC 0.9980 (K3) against the bfp4 reference; M covered
  about 80-512.

### The fixed cost and the crossover, diagnosed; 64-row sub-blocks from layout.rows_hint (2026-10-22, bh-lb-09, flat alone)
Same bench as the sweep above (test_flat_balanced_sweep.py, Galaxy 8x4 experts per chip, defaults); scripts and logs
/localdev/mstaletovic/flat_fix/ (run.sh, batch.sh, prof.py / tl.py / hlat.py: per-core zone timelines from
profile_log_device.csv with MIMO_FL_ZONES=1 / MIMO_FL_WAITZ).

**The "fixed cost" is mostly not fixed.** Scaling experts per call at one M separates the two (us per call, E 4 / 8 /
16 for GLM-5.3, 2 / 4 / 8 for M3):
| | GLM-5.3 M 32 | GLM-5.3 M 384 | M3 M 32 | M3 M 384 |
|---|---|---|---|---|
| calls | 220 / 408 / 797 | 442 / 789 / 1464 | 160 / 297 / 565 | 334 / 556 / 1021 |
| per expert (slope) / its floor | 48.6 / 43.0 | 84.4 / 60 DRAM, 58 math | 67 / 63.7 | 116 / 81 DRAM, 87 math |
| intercept (truly per call) | ~19 | ~114 | ~29 | ~92 |
At M 32 the gap is the stream rate (88-95% of the floor per expert) plus ~20-30 us per call; at M 384 it is the
steady state and a ~100 us fill + drain.

Where the per-call part goes (GLM-5.3 E 8, profiles):
- **The slowest DRISC slot sets the end.** Expert e's gate/up weights land every 44-46 us at M 32 (the first 58: it
  shares DRAM with two experts of down weights and x); per gate/up core the last block lands 429-460 us at M 160 - a
  7% spread between slots over a call, and every h waits for all 64 gate/up cores.
- **The drain after the last weights** (M 160, 128-row sub-blocks): last weights 440 / 460 us (median / slowest) ->
  last gate/up row pass 495 / 507 -> h at the down cores ~501 -> down done 527: 67-87 us = the last expert's remaining
  row passes (~26 us), the h hand-off (~6-15), the last 128-row down sub-block (20) and its 32-row tail (5).
- **The fill at M 384**: the first sub-block's K loop waits for expert 0's whole gate/up slice (lands at 67 us vs ~45
  in steady state); a first h at the down cores at ~105-111 us.

What the crossover's steady state waits on (M 384, E 8 / E 3):
- **Gate/up compute is the bound, not the DRAM / math overlap.** fp32 DST (4 tiles a half) forces 2-row-tile passes
  (row passes, SE_GU_RP): a pass of 64 rows x 2 tiles x K 192 takes 12.8 us (~71% of the LoFi tile rate), 25.6 us per
  128-row sub-block; the down cores take 20 us per sub-block (~91%) and wait ~7 us each for h. Gate/up also waits
  ~3.5 us per sub-block on x / weights (WAITZ): the x ring holds exactly one sub-block (x_slots 24 = nk_gu) and row
  passes keep it until the final pass, so the next sub-block's x streams in only during that pass.
- **DRAM is a near tie**: with x read by both rectangles (2 x 4.7 MB at M 384) the true bytes are ~35 MB an expert
  (~69 us); without the down-weight reads (MIMO_FL_DN_NOREAD, garbage) 790 -> 728 us. The two near-equal limits are
  why the time looks like a sum.

Tried, no gain (GLM-5.3 E 8 unless noted, us at M 32 / 160 / 384 / 1024 vs 408 / 539 / 790 / 1722):
- MIMO_FL_GU_FULL_SYNC (one 4-row pass in a full-sync DST): 408 / - / 770 / 1760; M3 and K3 +12-23% at M >= 384.
- h chains: DCH 4 / 16 / 32, HPIECES 8, LINK_DEPTH 8: within 0.5% (DCH 32 +6% at 384).
- down weights: DW_BATCH 1 / 4 unchanged; their start delayed 20-70 us at the call start (a probe, reverted):
  unchanged - the DN_NOREAD gain is steady-state sharing, not the fill.
- LEAN 3 + NS_MAX 10: 402 / 535 / - (-1.5% at M 32 only); M3 unchanged.
- x read once (X1): 411 / 755 / 1119 / 2644 (one primary's multicast caps x); M3 likewise.
- x ring 26 slots (MIMO_FL_X_SLOTS_MAX): **hangs** (gate/up waits on x at se3_compute.cpp:437: the row-pass x
  indexing assumes x_slots == nk_gu); 28 does not fit. Not pursued.
- 64-row sub-blocks with a 4th h buffer (MIMO_FL_HBUF=4 extended to bfp8 h, a probe, reverted): same as without it.

**Change: layout.rows_hint** (FlatLayoutOptions; expected mean rows per local expert per call, -1 unknown; in the
plan key and the plan log line). The capacity callers pass is the whole chunk, so the plan cannot infer M. With a
hint <= 320, plans that have 128-row sub-blocks and gate/up row passes (GLM-5.3, GLM-5.3-Flash; every other model
already plans 64-row ones) take 64-row sub-blocks: one row pass a sub-block (x streams in during the whole pass; the
drain is halved). y bit-exact (GLM-5.3 M 128). Above ~384 rows the 128-row sub-blocks are better (64-row ones:
+0.4% at 384, +3.1% at 512, +5.9% at 1024 on GLM-5.3), hence the threshold. The close-out tests pass the hint
(round(chunk x top-k / experts)): GLM-5.3 / GLM-Flash at 5120 take 64-row sub-blocks, at 16384 128-row ones.
test_flat_balanced_sweep.py: GLM_SWEEP_ROWS_HINT=<n> | auto (auto: the op rebuilt per M with the hint M).

Full sweep, before (the table above) -> after (hint = M; us / DRAM %):
| M | GLM-5.3 | GLM-5.3-Flash | K3 | K2.7 | M3 | DSv4.1-Flash |
|---|---|---|---|---|---|---|
| 32 | 408 -> 403 / 85 | 312 -> 307 / 84 | 1110 -> 1109 | 695 -> 696 | 297 -> 297 | 530 -> 529 |
| 96 | 468 -> 458 / 80 | 356 -> 347 / 80 | 1233 -> 1230 | 784 -> 782 | 354 -> 355 | 609 -> 608 |
| 128 | 516 -> 497 / 77 | 389 -> 377 / 76 | 1308 -> 1311 | 829 -> 829 | 375 -> 375 | 656 -> 658 |
| 160 | 539 -> 524 / 75 | 410 -> 397 / 74 | 1434 -> 1448 | 875 -> 877 | 390 -> 389 | 711 -> 716 |
| 192 | 572 -> 553 / 73 | 435 -> 420 / 72 | 1503 -> 1515 | 927 -> 925 | 411 -> 413 | 761 -> 762 |
| 256 | 642 -> 630 / 68 | 494 -> 479 / 67 | 1692 -> 1699 | 1062 -> 1057 | 457 -> 458 | 880 -> 871 |
| 384 | 787 -> 786 | 598 -> 599 | 2244 -> 2241 | 1350 -> 1345 | 555 -> 556 | 1088 -> 1086 |
| 1024 | 1722 -> 1722 | 1283 -> 1283 | 4999 -> 4999 | 3138 -> 3136 | 1159 -> 1156 | 2589 -> 2589 |
GLM-5.3 -1.2 .. -3.7% and GLM-Flash -1.5 .. -3.4% at M 32-256, unchanged above. The other plans are unchanged (their
plan is identical); the largest move, K3 at 160 / 192 (+0.9 / +0.8%), is run-to-run noise: HEAD without the change
gave 1441 / 1431 / 1448 at 160 and 1498 / 1519 / 1508 at 192, with it 1447 / 1440 / 1440 and 1515 / 1514 / 1518.
Correctness: test_closeout_flat 9 / 9 (min PCC 0.9980), test_flat_combine_closeout 9 / 9 under uniform, zipf
spread and zipf packed routing (flat min PCC 0.9984, worst per-token rel err 2.43e-3), test_closeout_plans passes.

Left (largest first): gate/up compute at mid / large M (2-row passes at ~71% of LoFi: a 4-row pass needs 8 fp32 DST
tiles - full sync lost the pack overlap; an x ring deeper than one sub-block needs the row-pass x indexing fixed);
the DRISC slot spread (~7% of a call, the slowest slot gates the drain); x read twice (~7% of DRAM bytes at M 160+);
the remaining drain (h hand-off + the last down sub-block).

### Gate/up K loop microbenchmark: where the "~71% of LoFi" goes (2026-10-22)
`tests/microbench/test_gu_matmul_bench.py`: one Tensix core, x bfp8 [RT x K] and gate/up weights bfp4 [K x CT]
resident in L1, LoFi, K 192 (GLM-5.3), 32 passes of one RT x CT block over all of K with stock `matmul_block` (one
call per K tile, as se3_compute), timed on the device wall clock (UNPACK after the inputs are resident -> PACK after
the last pack; counted as AICLK cycles, us at 1.35 GHz). Peak: 2 x 32^3 FLOP / 4096 FLOP per cycle = 16 cycles a tile
product. "stream" = CB wait / pop of x and w per K-block (the reader re-pushes resident blocks: the flat op's
handshakes without data movement); "act" = silu(gate) * up on the PACK thread's SFPU as act_dst. PCC vs torch on the
quantized inputs: fp32 1.0000, bf16 DEST 0.9989 (0.9967 with act).
| block | resident | resident + act | stream kblk 8 | stream + act |
|---|---|---|---|---|
| 2 x 2 fp32 (GLM gate/up pass) | 17.80 cyc (89.9%) | 18.04 (88.7%) | 19.75 (81.0%) | 20.00 (80.0%) |
| 1 x 3 fp32 (K3 / M3 / DSv4.1 pass) | 17.36 (92.2%) | 17.69 (90.4%) | 19.56 (81.8%) | 19.87 (80.5%) |
| 4 x 2 bf16 DEST (8-tile half) | 17.26 (92.7%) | 17.77 (90.0%) | 18.09 (88.4%) | 18.58 (86.1%) |
K-block size, stream, fp32: 2 x 2 kblk 4 / 8 / 16 / 24: 21.33 / 19.75 / 18.86 / 18.49 (75.0 / 81.0 / 84.8 / 86.5%);
1 x 3: 20.95 / 19.56 / 18.78 / 18.52. Shapes, resident fp32: 1 x 1 27.24 (58.7%), 1 x 4 17.27, 4 x 1 17.28 (92.6%).
- The FPU loop itself is ~90-93%: stock matmul_block saturates at ~17.3 cycles a product once 3+ output tiles share
  an operand (1 x 1 pays ~11 cycles a call). 2 x 2 is 0.5 cycle worse than 1 x 4.
- The per-K-block CB handshake (wait + pop of x and w) costs ~55-65 cycles a block: at kblk 8 on a 2 x 2 pass that is
  -9 points (90 -> 81%). Larger K-blocks (24: 86.5%) or larger blocks (4 x 2 bf16: 88.4%) amortize it.
- The activation on the PACK thread stays hidden (+1-3%).
- In the op the pass runs at ~22.5 cycles (71%): 17.8 loop + ~2 handshakes + ~2.7 around the loop (x / weight arrival
  waits, row passes, h hand-off).
- custom_mm-style loop (variant 4) not built: for full 32x32 tiles the stock LLK already has custom_mm's structure
  (one operand held in SrcB, in1 streamed through SrcA with CLR_A per tile, a 16-MVMUL replay per tile); what is left is
  folding the per-tile software step (dest base + MOP run) into one MOP over CT, bounded by the measured ~1.3 cycles a
  product (<= ~7.5%). custom_mm's other tricks (split accumulation and dense packing in the empty rows of <= 8-row
  tiles) do not apply to 32-row tiles. Not worth building into se3_compute ahead of the handshakes and the stalls.

### Bigger gate/up compute K-blocks: decoupled K groups vs larger DRISC blocks (2026-10-22)
Why: the gate/up microbench (above) put ~2 cycles per tile product in the per-K-block CB handshake (wait / pop of x and
weights, ~55-65 cycles a block of 8 K tiles): streamed K-block 8 / 16 / 24 = 81 / 85 / 86.5% of LoFi vs 90% resident.
- **A, decoupled** (se3_compute row-pass loop, `SE_GU_KG`): wait for / pop G blocks at once, the matmul still walks them
  block by block (same accumulation order: y bit-exact vs G 1 at GLM-5.3 M 128, G 2 and 3). DRISC blocks unchanged; the
  x ring (a whole sub-block) and the weight ring (2 loads) already hold G blocks. Typed `layout.gu_kgroup` (-1: 2), plan
  `gu_kg` (logged with the plan: "gate/up row passes .. K group .."), probe `MIMO_FL_GU_KG` overrides.
- **B, larger DRISC blocks**, emulated as A + `MIMO_FL_DRISC_UB=G` (each DRISC unit carries G blocks, the consumer pops G):
  the full form (KBLK beyond 8 in every kernel: relays, receivers, DRISC chunks, h layout) was not built because the
  emulation already loses.
us per call, Galaxy experts per chip, rows_hint = M (E 8 GLM-5.3, E 28 K3):
| variant | GLM-5.3 M 32 / 128 / 160 / 384 / 1024 | K3 M 32 / 160 / 384 / 1024 |
|---|---|---|
| today (G 1) | 403 / 498 / 523 / 788 / 1723 | 1111 / 1439 / 2247 / 5000 |
| A, G 2 | 403 / 499 / 523 / 784 / 1677 | 1111 / 1441 / 2233 / 5001 |
| A, G 3 | 404 / 500 / 524 / 786 / 1704 | - |
| A, G 4 | - / - / - / 786 / 1671 | 1111 / 1439 / 2253 / 4987 |
| B (UB 2 + G 2) | 389 / 500 / 569 / 847 / 1738 | 1111 / 1593 / 2419 / 5111 |
| B + LEAN 3 | 394 / - / 566 / 846 / 1722 | - |
B loses 7-11% at M 160-384 (the shallower DRISC pipeline next to x, as UB 2 alone did) and only wins GLM M 32 (-3.5%).
A: a few % at M >= 1024, nothing below 512 (there the handshake is not the bound). Full 66-point sweep, G 4: GLM-5.3
-2.6 / -2.8% at 1024 / 2048, DSv4.1 -1.0..-2.1% from 512, but M3 +1.7% at 2048 (rerun: +1.6%); G 2 (default):
| model | M 512 / 1024 / 2048 vs G 1 (us) | rest |
|---|---|---|
| GLM-5.3 | 967->960 / 1722->1676 / 3300->3188 (-0.7 / -2.7 / -3.4%) | within +-0.5% |
| GLM-5.3-Flash | 717->721 / 1283->1294 / 2462->2487 (+0.6 / +0.9 / +1.0%) | within +-0.5% |
| K3 | 2792->2785 / 4999->5006 / 9855->9854 | within +-0.6% |
| K2.7 | flat | within +-0.4% |
| M3 | 662->664 / 1156->1161 / 2131->2135 | within +-0.7% |
| DSv4.1-Flash | 1344->1330 / 2589->2556 / 5067->5002 (-1.0 / -1.3 / -1.3%) | within +-0.7% |
GLM-5.3-Flash at large M is noisy: a rerun gave G 1 / 2 / 4 = 1283 / 1290 / 1342 at 1024 and 2462 / 2476 / 2426 at 2048
(the earlier G 4 sweep: 1266 / 2405) - G 2 is within that noise. Correctness: test_closeout_flat 9/9, test_flat_combine_
closeout 9/9, test_closeout_plans pass (G 2 in every row-pass plan). Logs: /localdev/mstaletovic/gu_kblk/.
Takeaway: the handshake is a real but small cost in the op (the microbench's 81 -> 86% does not carry over: below M 512
the gate/up loop is not the bound, above it G 2 buys ~3% on the pair-split GLM-5.3 and ~1.3% on DSv4.1's units).

### GLM-5.3 alone at its Galaxy 8x4 load: what holds it at ~67% DRAM (2026-10-25, bh-lb-09, one chip)
Why: the Galaxy reports ~67% DRAM for GLM-5.3's flat expert (8 experts a chip, 160 home tokens, ~160 rows an expert at
chunk 5120); our 2x4 numbers said the same. test_flat_balanced_sweep, E 8, H 6144, I 2048, silu, bfp4, rows_hint = M,
DRISC-only defaults (pure, no Tensix readers), even split (64 gate/up cores x 1 pair, 32 down x 6 tiles, 4 relays).
Weights 21.2 MB an expert (3 x 6144 x 2048 x 0.5625 B) = 41.5 us at 512 GB/s; x + y bf16 1.97 MB each an expert.
| M | M_total | total us | us / expert | DRAM % (w + x + y) | DRAM % (weights) | math % (96 compute) | math % (110 grid) |
|---|---|---|---|---|---|---|---|
| 96 | 768 | 458 | 57.2 | 81 | 72 | 23 | 21 |
| 128 | 1024 | 498 | 62.3 | 76 | 67 | 29 | 26 |
| 160 | 1280 | 521 | 65.2 | 75 | 64 | 35 | 30 |
| 192 | 1536 | 554 | 69.3 | 73 | 60 | 39 | 34 |
| 224 | 1792 | 599 | 74.9 | 70 | 55 | 42 | 37 |
| 256 | 2048 | 631 | 78.8 | 68 | 53 | 46 | 40 |
Galaxy loads of the others: GLM-5.3-Flash E 9 M 142: 381 us, 42.3 / expert, 76% DRAM; K3 E 28 M 91 (H 3584, I 3072,
units): 1233 us, 44.0 / expert, 88%. The gap over the DRAM floor (w + x + y) is ~90-150 us a call on all three: K3
hides it under 28 experts' weights, GLM-5.3's 8 do not.

Probes at M 160 (us, base 521; one-chip, MIMO_FL_* probes):
- No math: 527 - the compute is not the bound (gate/up's math hides under the rest).
- DRAM reads: any one stream off - x (XRD_SKIP) 442, gate/up W (GU_NOREAD / W_NOREAD) 434, down W (DN_NOREAD) 433 -
  or all three 434: the read streams interfere (~87 us), removing any one removes it (the x reads go through each
  bank's NOC1 endpoint = DRISC tile 1's port: drisc-endpoint-tile-x-conflict).
- Below that an on-chip floor: no reads 434, + no math 414, + no weight forwards (W_NOFWD) 334, + no x / h / y data
  333: the control skeleton (every handshake, no data, no math) runs 333 us. Weight forwards into the gate/up L1 ~81.
- The skeleton per (expert, sub-block): GLM-5.3 28.6 us at M 96 (2 sub-blocks of 64 rows: 24 K-blocks each),
  41.6 at M 160 (3 sub-blocks), ~1.2 us a K-block per 128 rows - the same on GLM-5.3-Flash (19.7 us, 16 K-blocks) and
  K3 (16.0, 14): the op's per-step cost. At H 6144 it about equals the weights' DRAM time (41.5 us an expert), so the
  two run near serially. Sampled gate/up waits (WAITZ, K group 1): x 3.6 us, weights 0.5 a 28 us sub-block - the loop
  is not starved; K group 1 / 2 / 4 / 8 / 12 / 24: 334 / - / 328 / 330 / 332 / 351 (the compute handshake is not it).
- Not levers here: x helpers 2 (6 relays) 515, 128-row sub-blocks (MT 4) 535, + full-sync gate/up 527, GU_SPLIT 2,2
  (2 M-groups) 701, 4,4 (-> 2 units) 526, 2,1 (32 gate/up cores x 4 tiles: 18.4 KB blocks) 642 (M 96 61.6 vs 57.2),
  DRISC NOC1 0 521, JOB1_NOC 0 / 1 524 / 525, STREAM_SPLIT 0 525, TILES 2 530, TILES 1 565.
- X_RESIDENT hung (probe, twice; triage in /localdev/mstaletovic/flat_glm53/hang_x_resident) - not chased.
Levers (estimates, M 160):
- The per-K-block control cost (~1.2 us / K-block / 128 rows): fewer, larger steps end to end (16-tile K-blocks with
  a ring as deep as today's in bytes - the 2026-10-22 UB 2 emulation lost from a shallower DRISC pipeline): skeleton
  42 -> ~21 us / expert; with forwards + interference on top of the DRAM floor ~430-450 us (82-88%).
- The read interference (~87 us): x off the DRISC tile's port (x landed in the relays' L1 by the dispatch, or x reads
  on NOC0 once the linked-multicast hang is fixed): -50..-87 us.
- Sub-blocks sized to M (5 row tiles as 3 + 2, not 2 + 2 + 1): the skeleton's third sub-block is ~13 us / expert, but
  128-row sub-blocks did not show it in the real run (535) - low.
Logs: /localdev/mstaletovic/flat_glm53/.

### Lever 1 (fewer, larger K steps): emulations say it is not the bound (2026-10-25, bh-lb-09, GLM-5.3 E 8)
Why: the GLM-5.3 analysis above priced the per-K-block control cost at ~1.2 us / K-block / 128 rows and proposed
16-tile K-blocks end to end. A global kKBlk = 16 hangs at the first call (se6_drecv / se5_recv / se11_xrd in
cb_reserve_back: every kernel assumes 8-tile blocks; triage /localdev/mstaletovic/flat_glm53/hang_k16; reverted).
Before rewriting every kernel, each step type was made larger on its own with the existing knobs (us a call, M 96 /
128 / 160; base 459 / 500 / 522):
| variant (one variable) | real run | no reads (XRD_SKIP GU_NOREAD DN_NOREAD; base 386 / - / 436) | skeleton M 160 (base 333) |
|---|---|---|---|
| DRISC units of 2 blocks (DRISC_UB 2: half the DRISC loop's per-unit cost) | 468 / 497 / 571 | 381 / - / 437 | - |
| x pieces 2 (8.7 KB snake hops, half the handshakes) | 461 / 503 / 526 | 380 / - / 424 | 314 |
| x pieces 1 | 474 / 510 / 538 | - | 328 |
| x pieces 8 | - | - | 594 |
| snake forward depth 6 (vs 3) | 459 / 499 / 524 | - | 313 |
| pieces 2 + depth 6 | 463 / 502 / 526 | - | - |
| gate/up K group 4 (one compute handshake per 32 K tiles) | - | 389 / - / 434 | (K group 1..24 earlier: 328-351) |
| x ring 32 / 48 blocks (MT 2; 24 = one sub-block) | 457 / 497 / 527 ; 457 / 496 / 524 | - | 2 snakes a rectangle: 325 |
- DRISC profile at M 160 (MIMO_FL_DRISC_PROF): each tile ~1000 cycles a 9.2 KB unit, busy 681-792 k cycles of a
  ~860 k-cycle call; UB 2 cuts the busy time to ~535 k for the same call time: the DRISC loop is not what paces.
- Skeleton zones (M 160): the gate/up UNPACK thread waits in its K loop 316 of 321 us (math 1.7, pack 0.8), the down
  cores wait for h, the relays' readers wait on a full ring (XRD_FULL 195 us): the gate/up cores are starved of x.
  The skeleton tracks the padded x bytes a gate/up core takes in: M 96 / 160 / 256 = 2 / 3 / 4 sub-blocks of 64 rows
  = 6.7 / 10.0 / 13.4 MB a core a call at ~32 GB/s = 209 / 313 / 418 us vs 229 / 333 / 436 measured; bf16 x (1.9x
  the bytes, its own plan) 229 -> 332 / 333 -> 476. Fewer handshakes (pieces 2, depth 6, K group) move it <= 6%.
Takeaway: no larger-step emulation moves the real run by more than noise (best -0.5%), the no-read floor by <= 3%;
the 16-tile rewrite of every kernel is not worth it on this evidence. The real run is the on-chip floor (no reads
434) + the read interference (~87 us): lever 2 next. (The "x bytes" reading of the skeleton is wrong - see the
padding section below: the skeleton is a fixed ~0.54 us per x block whatever its bytes.)
Logs: /localdev/mstaletovic/flat_glm53/ (ab_*, r_*, nr*, xs_*, zs_sk, zw_sk, sk_xbf16, ub2_glm53, dprof).

### Lever 2 (x off the DRISC tile's port): retest with the unicast snakes (2026-10-25, bh-lb-09, GLM-5.3 E 8)
x path today: the 4 relays (BRISC) gather x rows from the row-major bf16 dispatch buffer in DRAM (2 KB segments, NOC1
reads -> each bank's NOC1 endpoint = DRISC tile 1, which also DMAs half the bank's gate/up weights in switch mode),
tilize to bfp8 (TRISC), and their NCRISC unicasts each super-block to one snake head per gate/up rectangle; the
gate/up cores forward it down the snake (NCRISC, NOC0, 4 pieces, 3 in flight). No multicast, fabric not involved.
Ports: tile 0 (NOC0 endpoint) = the down cores' weight reads; tile 1 (NOC1 endpoint) = x reads + DRISC job; tile 2 =
the other DRISC job. One variable a run, us a call M 96 / 128 / 160 (base 459 / 500 / 522), no hang in any:
| variant | us | note |
|---|---|---|
| x reads on NOC0 (MIMO_FL_XRD_NOC=0: sender to NOC1) | 470 / 499 / 546 | y bit-exact vs base; x on tile 0 beside the down weights |
| down weight reads on NOC1 (DN_NOC0=1: down weights on tile 1 beside the DRISC job and x) | 508 / 545 / 615 | |
| both: x alone on tile 0's port, down weights on tile 1 | 460 / 495 / 553 | x isolated from every DRISC port: no gain |
| x reads of 2 / 6 segments (XRD_BIG, same bytes) | 459 / 495 / 534 ; 503 / 565 / 611 | bigger requests hurt |
| XRD_BATCH 4 | 460 / 499 / 540 | |
| half the x bytes, same requests (XRD_HALF, garbage x) | 430 / 462 / 481 | recovers 70 / 100 / 50% of x's cost |
| no x reads (XRD_SKIP) | 418 / 425 / 437 | x's cost: 41 / 75 / 85 us (its bytes: ~31 us at M 160) |
Takeaway: the port is not the mechanism - x on a port of its own (tile 0, down weights moved) costs as much as on
tile 1. x's cost follows its bytes (half the bytes, half the cost), i.e. GDDR contention from ~16 MB of 2 KB row
gathers a call next to the 170 MB weight streams. The lever left is fewer x bytes: block-fp8 rows from the dispatch
(one exponent per 16 consecutive columns = exactly the bfp8 tile's per-face-row groups, so no accuracy change vs
today's tilize; 1.0625 B / element) with a relay-side rearrange into bfp8 tiles: estimate -30..-40 us at M 160
(XRD_HALF), dispatch + relay work. Landing x in L1 instead does not fit: M_total 1280 x 6144 bf16 = 15.7 MB a chip.
Logs: /localdev/mstaletovic/flat_glm53/ (x0_glm53, l2_*).

### Padding rows: not the cost; the skeleton is a fixed rate per x block (2026-10-25, GLM-5.3 E 8)
Before sizing the last sub-block to its real rows (M 160: 64 + 64 + 32, the third ships 64 rows of x), the emulation:
32-row sub-blocks (MIMO_FL_MT=1: no padding at all, more sub-blocks) - skeleton M 96 / 128 / 160 313 / 410 / 508 (3 /
4 / 5 sub-blocks) vs 64-row 229 / 229 / 333 (2 / 2 / 3); real 472 / 509 / 572 vs 459 / 500 / 522. Every sub-block
costs the same ~100 us a call (12.5-13 us an expert) whether it has 1 or 2 row tiles: shipping fewer rows saves
nothing, fewer sub-blocks does. Not implemented.
- The per-sub-block cost is 24 x blocks at ~0.54 us each at the gate/up core (X_ARR marks every 8 blocks 4.3 us apart,
  the unpacker in its K loop the rest), the same for 8.7 and 17.4 KB blocks.
- Skeleton M 96 / 160 exactly 229 / 333 with: 6 relays (NH 2), tilize skipped (TZ_SKIP), a 2-sub-block x ring (48),
  2 snakes a rectangle (325), forward depth 6 (313), pieces 2 / 1 (314 / 328); pieces 8 594. The relays are
  backpressured (XRD_FULL 195 us), the down cores wait for h, the weights land ahead (every 8 blocks ~6.4 us).
  What fixes the 0.54 us a block is not found yet - none of the x path's own knobs.
Logs: /localdev/mstaletovic/flat_glm53/ (pad_*, sk_nh2, sk_tzskip, sk_xs48, zs_sk2 + zs_sk2.csv).
- What sets the 0.54 us (per-core X_ARR marks, zs_sk2.csv): 2 snakes of 32 gate/up cores each, every core gets blocks
  at the same 0.54 us, ~0.43 us later than its predecessor; the head runs a full x ring (24 blocks, 12.5 us) ahead of
  core 2. It is the NCRISC forwarder's acked writes: 3 pieces in flight, one ack round trip (~0.4 us) each ->
  4 pieces = 0.53 us a block whatever the block's bytes (MT 1 / MT 2 the same). Pieces 8 predicts 1.07 us a block,
  measured 1.03 (594); forward depth 12: skeleton 333 -> 304 (M 96 229 -> 217), where the next limit is the snake's
  bandwidth, ~35-40 GB/s a link: 128-row sub-blocks (MT 4, 34.8 KB blocks) take ~0.9 us a block, so 2 x 24 x 0.9 ~
  3 x 24 x 0.54 (skeleton MT 4 M 160 345 vs 333) - why MT 4 did not help (real 535, no math 531).
- And it does not reach the real run: depth 12 no reads 387 / 434 (= base), real 459 / 499 / 526. The skeleton is not
  what binds; above it sit the weight forwards (+80 us at no reads) and the read interference (+87). Not fixed: a
  posted-write forwarder (no ack round trip) is estimated at ~0 in the real run on this evidence.
Logs: sk_mt4, sk_p2d6, sk_p4d12, nm_mt4, r_d12, nr_d12.

### Forward depth 12 as a default? No: +0.3..0.8% at the larger Galaxy M (2026-10-25, bh-lb-09, one chip)
us a call, base -> MIMO_FL_X_FWD_DEPTH=12: GLM-5.3 E 8 M 96 / 128 / 160 459 / 500 / 522 -> 459 / 499 / 526; GLM-5.3-
Flash E 9 M 96 / 142 / 192 347 / 381 / 417 -> 347 / 383 / 420; K3 E 28 (situ) M 64 / 91 / 128 1171 / 1233 / 1314 ->
1170 / 1235 / 1307; K2.7 E 12 M 80 / 107 / 160 757 / 807 / 874 -> 759 / 809 / 877. Repeat-run noise here is ~1 us
(GLM-5.3 M 160 base 521 / 522 / 522), so +3-4 us at GLM-5.3 160 / GLM-Flash 192 / K2.7 160 is real if small; only K3
M 128 gains (-0.5%). Default stays 3 (the skeleton's -29 us does not reach the real run).

### The weight forwards (DRISC -> gate/up L1): what the +80 us at no reads is (2026-10-25, GLM-5.3 E 8)
At no reads (XRD_SKIP GU_NOREAD DN_NOREAD; 386 / 436 us at M 96 / 160) the forwards cost 137 / 83 us (W_NOFWD 249 /
353), but in the real run only 15 / 5 (W_NOFWD with reads 444 / 517 vs 459 / 522).
| no reads, one variable | M 96 / 160 |
|---|---|
| base | 386 / 436 |
| no forwards (W_NOFWD) | 249 / 353 |
| DRISC 14 units in flight / LEAN 1 / free slots once sent | 384 / 433 ; 384 / 435 ; 383 / 434 |
| DRISC job 1 on NOC1 / on NOC0 | 381 / 429 ; 386 / 435 |
| DRISC NOC1 0 (tile 2's half on NOC0 too) | 412 / 473 |
| x snake on NOC1 (MIMO_FL_X_CHAIN_NOC=1) | 342 / 376 |
| real run, x snake on NOC1 (y bit-exact) / + x reads on NOC0 | 457 / 495 / 530 ; 471 / 503 / 546 (M 96 / 128 / 160) |
- The DRISC loop is not the cost (its in-flight / register / free-on-sent knobs change nothing); the weight writes slow
  the x snake where they share NOC0 links (more weights on NOC0: +26 / +37; the snake moved to NOC1: -44 / -60).
- With the DRAM reads on, that recovery vanishes (the snake on NOC1 meets the x reads' NOC1 traffic) and the forwards
  themselves cost <= 15 us: not a real-run lever. The real run is bound by the three read streams together (any one
  off: ~434 at M 160, DRAM 75% with all three); the on-chip side only shows once one stream is gone.
Logs: dd_*, wf_*, r_xnoc1*.

### Read mix per bank: MPFE port weights; down weights through the DRISC (2026-10-25, GLM-5.3 E 8)
MIMO_FL_DRISC_MPFE=w,o,x (the DRISC port, NOC0 endpoint = down reads, NOC1 endpoint = x reads), us M 96 / 128 / 160
(base 459 / 500 / 522), no hang: 0,0,3 461 / 490 / 526; 0,3,0 459 / 502 / 527; 3,0,0 463 / 499 / 530; 0,3,3 459 /
491 / 528. Only M 128 gains (-2%, x's port favoured); M 160 +1%. Not a lever.
Down weights through the DRISC: there is no such path (MIMO_FL_DRISC_RD_DOWN only reuses freed reader cells as down
cores). Not worth building: a DRISC tile forwards ~12-16 GB/s in this op (DRISC_PROF ~1000 cycles a 9.2 KB unit, the
~28 GB/s ceiling at best), so all 21.2 MB of an expert over the 16 tiles is >= ~50-88 us an expert against 65 today;
the ceiling of any change to the down stream is DN_NOREAD 433 at M 160.
So GLM-5.3 E 8 sits at ~75% DRAM with three read streams per bank; every on-chip and port knob tried moves it < 2%.
The measured lever left is fewer x bytes (block-fp8 x rows, -30..-40 us at M 160; parked by the user: no quantization).
Logs: rm_*.

### Flat vs main's routed experts (unified, fused, hybrid) at the Galaxy per-chip loads (2026-10-25, one chip, bh-lb-09)
Main = origin/main 07fac6de536 (worktree /localdev/mstaletovic/tt-metal-main-1025, built as is). Every implementation
on the same bfp4 experts at LoFi; device time from the program real-time profiler, median of 10 calls, every program
of a call summed. Flat: test_flat_balanced_sweep (DRISC-only defaults, rows_hint = M, x bf16 row-major gathered by
token index, y bf16 row-major, fp32 DEST down). Main: scratch harness zz_re_compare.py (copy in
/localdev/mstaletovic/re_cmp/) on TtRoutedExpert's weights and its production config (LoFi, bf16 DEST, packer L1 acc):
row-major bf16 dispatch buffer, E regions of 512 rows; "unified" = unified_routed_expert_moe; "fused" = TtRoutedExpert
.forward with T >= max_tokens (moe_fused_swiglu alone); "hybrid op" = the one-dispatch hybrid_routed_expert_moe at
T 320 (the models' ROUTED_EXPERT_HYBRID_TOKEN_THRESHOLD) and T 128; skewed: zipf (count ~ 1 / rank, same total, cap
512), also the two-op forward at T 128. DRAM % = (weights + x bf16 + y; y bf16 flat, bfp8 main) / 512 GB/s. Math %
over the compute cores (flat 64 gate/up + 32 down, K3 + 28; main 11 x 8 = 88) and over 110. * = the Galaxy M.

**GLM-5.3 E8** (Galaxy M 160), cells: total us / us per expert / DRAM % / math % compute cores / math % 110 grid; flat compute = 96 cores, main's ops 88 (11 x 8)

| M | flat | unified | fused (moe_fused_swiglu) | hybrid op T 320 | hybrid op T 128 | best main / flat |
|---|---|---|---|---|---|---|
| 64 | 433 / 54.1 / 82% / 17% / 15% | 697 / 87.1 / 50% / 11% / 9% | 533 / 66.6 / 66% / 15% / 12% | 562 / 70.2 / 62% / 14% / 11% | 561 / 70.1 / 63% / 14% / 11% | 1.23x |
| 96 | 459 / 57.4 / 80% / 24% / 21% | 694 / 86.7 / 52% / 18% / 14% | 561 / 70.1 / 64% / 21% / 17% | 604 / 75.5 / 60% / 20% / 16% | 602 / 75.3 / 60% / 20% / 16% | 1.22x |
| 128 | 498 / 62.2 / 77% / 30% / 26% | 691 / 86.3 / 53% / 22% / 18% | 589 / 73.6 / 63% / 28% / 22% | 626 / 78.3 / 59% / 25% / 20% | 623 / 77.9 / 59% / 25% / 20% | 1.18x |
| 160 * | 526 / 65.8 / 75% / 34% / 30% | 711 / 88.8 / 53% / 28% / 22% | 698 / 87.3 / 54% / 29% / 23% | 744 / 93.0 / 51% / 26% / 21% | 818 / 102.2 / 46% / 24% / 19% | 1.33x |
| 192 | 555 / 69.4 / 73% / 39% / 34% | 733 / 91.6 / 53% / 32% / 26% | 711 / 88.9 / 55% / 34% / 27% | 746 / 93.3 / 52% / 32% / 26% | 829 / 103.7 / 47% / 29% / 23% | 1.28x |
| 256 | 632 / 79.0 / 68% / 46% / 40% | 757 / 94.6 / 54% / 42% / 34% | 753 / 94.1 / 54% / 42% / 34% | 765 / 95.6 / 53% / 41% / 33% | 853 / 106.7 / 48% / 38% / 30% | 1.19x |
| zipf (mean 160) | 529 / 66.1 / 74% / 34% / 30% | 748 / 93.5 / 51% / 26% / 21% | 697 / 87.1 / 54% / 29% / 23% | 711 / 88.9 / 53% / 28% / 22% | 731 / 91.3 / 52% / 28% / 22% | 1.28x (two-op T 128: 679) |

**GLM-5.3-Flash E9** (Galaxy M 142), cells: total us / us per expert / DRAM % / math % compute cores / math % 110 grid; flat compute = 96 cores, main's ops 88 (11 x 8)

| M | flat | unified | fused (moe_fused_swiglu) | hybrid op T 320 | hybrid op T 128 | best main / flat |
|---|---|---|---|---|---|---|
| 64 | 324 / 36.0 / 83% / 17% / 15% | 591 / 65.7 / 44% / 10% / 8% | 444 / 49.3 / 59% / 14% / 11% | 489 / 54.3 / 54% / 12% / 10% | 488 / 54.2 / 54% / 12% / 10% | 1.37x |
| 96 | 349 / 38.8 / 79% / 23% / 20% | 590 / 65.6 / 46% / 15% / 12% | 489 / 54.4 / 55% / 19% / 15% | 540 / 60.0 / 50% / 16% / 13% | 539 / 59.9 / 50% / 16% / 13% | 1.40x |
| 128 | 375 / 41.7 / 76% / 29% / 25% | 580 / 64.4 / 48% / 20% / 16% | 494 / 54.9 / 56% / 24% / 19% | 558 / 62.0 / 50% / 21% / 17% | 557 / 61.8 / 50% / 21% / 17% | 1.32x |
| 142 * | 381 / 42.4 / 76% / 32% / 28% | 583 / 64.8 / 48% / 22% / 18% | 637 / 70.8 / 44% / 21% / 17% | 683 / 75.9 / 41% / 19% / 15% | 688 / 76.5 / 41% / 19% / 15% | 1.53x |
| 160 | 398 / 44.2 / 74% / 34% / 30% | 588 / 65.4 / 48% / 25% / 20% | 637 / 70.8 / 45% / 24% / 19% | 682 / 75.8 / 42% / 21% / 17% | 691 / 76.7 / 41% / 21% / 17% | 1.48x |
| 192 | 420 / 46.7 / 72% / 39% / 34% | 614 / 68.2 / 47% / 29% / 23% | 637 / 70.8 / 46% / 28% / 22% | 683 / 75.8 / 43% / 26% / 21% | 703 / 78.1 / 41% / 25% / 20% | 1.46x |
| 256 | 477 / 53.0 / 68% / 46% / 40% | 631 / 70.1 / 48% / 38% / 30% | 653 / 72.5 / 47% / 36% / 29% | 688 / 76.4 / 44% / 35% / 28% | 720 / 80.0 / 42% / 32% / 26% | 1.32x |
| zipf (mean 142) | 395 / 43.9 / 73% / 31% / 27% | 625 / 69.5 / 45% / 21% / 17% | 594 / 66.0 / 47% / 22% / 18% | 615 / 68.4 / 46% / 21% / 17% | 618 / 68.7 / 45% / 21% / 17% | 1.43x (two-op T 128: 564) |

**K3 E28 (H 3584)** (Galaxy M 91), cells: total us / us per expert / DRAM % / math % compute cores / math % 110 grid; flat compute = 92 cores, main's ops 88 (11 x 8)

| M | flat | unified | fused (moe_fused_swiglu) | hybrid op T 320 | hybrid op T 128 | best main / flat |
|---|---|---|---|---|---|---|
| 64 | 1170 / 41.8 / 91% / 20% / 17% | 2565 / 91.6 / 41% / 10% / 8% | 1840 / 65.7 / 57% / 14% / 11% | 1995 / 71.3 / 53% / 12% / 10% | 1991 / 71.1 / 53% / 12% / 10% | 1.57x |
| 91 * | 1231 / 44.0 / 88% / 26% / 22% | 2568 / 91.7 / 42% / 14% / 11% | 2187 / 78.1 / 49% / 16% / 13% | 2359 / 84.3 / 45% / 15% / 12% | 2354 / 84.1 / 45% / 15% / 12% | 1.78x |
| 96 | 1231 / 44.0 / 89% / 29% / 24% | 2569 / 91.7 / 42% / 14% / 11% | 2186 / 78.1 / 49% / 16% / 13% | 2358 / 84.2 / 46% / 15% / 12% | 2354 / 84.1 / 46% / 15% / 12% | 1.78x |
| 128 | 1313 / 46.9 / 85% / 36% / 30% | 2499 / 89.2 / 44% / 20% / 16% | 2273 / 81.2 / 48% / 21% / 17% | 2415 / 86.3 / 45% / 20% / 16% | 2413 / 86.2 / 45% / 20% / 16% | 1.73x |
| 160 | 1436 / 51.3 / 79% / 41% / 34% | 2493 / 89.0 / 45% / 25% / 20% | 3050 / 108.9 / 36% / 20% / 16% | 3337 / 119.2 / 33% / 19% / 15% | 2835 / 101.3 / 39% / 21% / 17% | 1.74x |
| 192 | 1508 / 53.9 / 77% / 47% / 39% | 2513 / 89.7 / 45% / 29% / 23% | 3051 / 109.0 / 37% / 24% / 19% | 3337 / 119.2 / 34% / 21% / 17% | 2820 / 100.7 / 40% / 26% / 21% | 1.67x |
| 256 | 1708 / 61.0 / 71% / 55% / 46% | 2632 / 94.0 / 44% / 38% / 30% | 3067 / 109.5 / 38% / 31% / 25% | 3339 / 119.3 / 35% / 29% / 23% | 2892 / 103.3 / 40% / 34% / 27% | 1.54x |
| zipf (mean 91) | 1274 / 45.5 / 85% / 26% / 22% | 2651 / 94.7 / 40% / 12% / 10% | 2269 / 81.0 / 47% / 15% / 12% | 2356 / 84.1 / 45% / 15% / 12% | 2294 / 81.9 / 47% / 15% / 12% | 1.68x (two-op T 128: 2139) |

**K2.7 E12** (Galaxy M 107), cells: total us / us per expert / DRAM % / math % compute cores / math % 110 grid; flat compute = 96 cores, main's ops 88 (11 x 8)

| M | flat | unified | fused (moe_fused_swiglu) | hybrid op T 320 | hybrid op T 128 | best main / flat |
|---|---|---|---|---|---|---|
| 64 | 733 / 61.1 / 85% / 17% / 15% | 1166 / 97.2 / 53% / 12% / 10% | 885 / 73.7 / 69% / 16% / 13% | 939 / 78.3 / 65% / 15% / 12% | 942 / 78.5 / 65% / 15% / 12% | 1.21x |
| 96 | 782 / 65.2 / 82% / 24% / 21% | 1157 / 96.4 / 54% / 18% / 14% | 935 / 77.9 / 67% / 22% / 18% | 992 / 82.6 / 64% / 21% / 17% | 991 / 82.6 / 64% / 21% / 17% | 1.20x |
| 107 * | 806 / 67.2 / 81% / 26% / 23% | 1163 / 96.9 / 55% / 20% / 16% | 991 / 82.5 / 64% / 24% / 19% | 1026 / 85.5 / 62% / 22% / 18% | 1028 / 85.7 / 62% / 22% / 18% | 1.23x |
| 128 | 829 / 69.1 / 80% / 31% / 27% | 1166 / 97.2 / 55% / 24% / 19% | 989 / 82.4 / 65% / 28% / 22% | 1026 / 85.5 / 63% / 28% / 22% | 1026 / 85.5 / 63% / 28% / 22% | 1.19x |
| 160 | 878 / 73.1 / 78% / 37% / 32% | 1197 / 99.8 / 55% / 29% / 23% | 1159 / 96.6 / 57% / 30% / 24% | 1208 / 100.7 / 55% / 29% / 23% | 1369 / 114.1 / 48% / 25% / 20% | 1.32x |
| 192 | 926 / 77.1 / 77% / 41% / 36% | 1239 / 103.3 / 55% / 34% / 27% | 1195 / 99.6 / 57% / 35% / 28% | 1211 / 100.9 / 56% / 35% / 28% | 1393 / 116.1 / 49% / 30% / 24% | 1.29x |
| 256 | 1059 / 88.2 / 71% / 48% / 42% | 1283 / 106.9 / 56% / 44% / 35% | 1282 / 106.8 / 56% / 44% / 35% | 1265 / 105.4 / 56% / 44% / 35% | 1439 / 119.9 / 50% / 39% / 31% | 1.19x |
| zipf (mean 107) | 814 / 67.8 / 80% / 26% / 23% | 1223 / 101.9 / 52% / 19% / 15% | 1056 / 88.0 / 60% / 22% / 18% | 1089 / 90.7 / 58% / 21% / 17% | 1119 / 93.3 / 57% / 21% / 17% | 1.28x (two-op T 128: 1044) |

- Flat is fastest at every point: best of main / flat 1.18-1.33x on GLM-5.3, 1.32-1.53x GLM-5.3-Flash, 1.54-1.78x K3,
  1.19-1.32x K2.7; zipf 1.28 / 1.43 / 1.68 / 1.28x. Flat holds 71-91% DRAM; main's best 44-69%.
- Among main's: the fused op wins at <= 128 tokens an expert, then steps up (GLM-5.3 589 -> 698 us at 128 -> 160,
  GLM-Flash 494 -> 637 at 142, K3 2273 -> 3050 at 160); unified is ~flat in M (87-107 us an expert) and wins above 128 on
  GLM-Flash / K3. The models' T 320 sends 129-320-token experts to the fused op, slower than unified there (GLM-Flash
  M 142 637 vs 583, K3 M 160 3050 vs 2493): ~128 is the crossover on these shapes.
- The one-dispatch hybrid op costs over the separate ops: T 320 (all fused) +5-9% over the fused op alone, its unified
  half (T 128 at M >= 160) +12-15% over unified alone. Under zipf routing the two-op forward at T 128 is main's best.
Logs: /localdev/mstaletovic/re_cmp/ (fl_*, mn_*, mz_*, main_all.out, flat_all.out, table.md).

### Fixed costs per call: what sits 90-150 us above the DRAM floor (2026-10-26, one chip bh-lb-09, DRISC-only defaults)
Why: at the Galaxy loads every model runs ~90-150 us above its DRAM floor (weights + x bf16 + y bf16 at 512 GB/s); K3
hides it under 28 experts, GLM-5.3 (8) cannot. Two methods: an E x M regression, and per-role zone timestamps.

**1. Regression.** test_flat_balanced_sweep, E 1 / 2 / 4 / 8 / 16 / 28 x M 32 / 64 / 96 / 128 / 160 / 256, uniform,
rows_hint = M (64-row sub-blocks at every point), 144 points, no hang. The overhead (time - floor) is linear in E at
each M (rms 1-5 us), with a per-call intercept a(M) and a per-expert slope b(M):
| M | GLM-5.3 a / b | GLM-5.3-Flash a / b | K3 a / b | K2.7 a / b |
|---|---|---|---|---|
| 32 | 18 / 5.7 | 18 / 3.8 | 26 / 1.6 | 18 / 6.7 |
| 64 | 36 / 4.8 | 28 / 3.1 | 35 / 2.5 | 39 / 5.8 |
| 96 | 41 / 6.1 | 32 / 4.2 | 45 / 3.4 | 46 / 7.0 |
| 128 | 49 / 7.8 | 38 / 5.4 | 51 / 5.0 | 53 / 8.9 |
| 160 | 53 / 9.6 | 44 / 6.5 | 52 / 8.9 | 60 / 10.9 |
| 256 | 59 / 17.2 | 49 / 12.0 | 62 / 15.2 | 69 / 19.4 |
(us per call / us per expert.) One global fit over M <= 160, a + b E + c E ceil(M / 64) + d E M, leaves rms 9-13 us
(max 22-32): a 32-43 us a call, b -2..2.5 an expert, c 1.0-1.6 an expert sub-block, d 19-39 ns an expert row. So:
the per-call part grows with M (18 -> 53-60 us from M 32 to 160), the per-expert part is 2-7 us at small M and grows
with M (the x / y traffic, then math above M ~160). Galaxy loads from the per-M fits: GLM-5.3 E 8 M 160 53 + 77 = 130
(measured 129), GLM-Flash E 9 M 142 ~41 + ~55, K3 E 28 M 91 ~43 + ~90, K2.7 E 12 M 107 ~48 + ~90.

**2. Pieces (zones, MIMO_FL_ZONES=1, the second of 2 calls; new receiver marks W_E0 / W_E1 = each expert's first /
last gate/up weight block landed on a core, X_FIRST = its first x block; K3 / K2.7 with the every-8-blocks marks off,
or the profiler buffer drops the tail).** us from the call's first kernel start; floor/expert = (w + x + y) / 512.
| | GLM-5.3 E 8 M 160 | GLM-Flash E 9 M 142 | K3 E 28 M 91 | K2.7 E 12 M 107 |
|---|---|---|---|---|
| call (profiled) / floor | 529 / 393 | 381 / 290 | 1234 / 1087 | 805 / 653 |
| first weight block landed (min / med) | 8.7 / 10.4 | 8.5 / 10.3 | 12.6 / 14.4 | 9.3 / 10.9 |
| first x block landed (med / max) | 15.6 / 22.6 | 15.8 / 22.9 | 14.9 / 22.1 | 16.4 / 24.0 |
| weight landing per expert on a core (med) / gap to the next expert | 47.2 / 3.0 | 31.3 / 3.1 | 36.0 / 3.9 | 54.6 / 2.5 |
| sum of the gaps (median core) | 45 (1.9-3 early, 6.5 / 11.7 / 16.5 at experts 4-6) | 28 | ~100 | 28 |
| last weight block landed (med / max) | 439 / 453 | 320 / 341 | 1164 / 1195 | 711 / 764 |
| last gate/up K loop end (max) | 512 | 365 | 1212 | 780 |
| kernel end (max: down cores, y writes) | 529 | 381 | 1234 | 805 |
Fixed-cost table (us at the Galaxy load):
| fixed cost | GLM-5.3 | GLM-Flash | K3 | K2.7 | scales with | lever |
|---|---|---|---|---|---|---|
| head: launch -> first weight block lands (DRAM idle) | 10 | 10 | 14 | 11 | nothing (K3 +4: the 28-expert schedule) | start the first expert's weight stream before the op (in the dispatch / previous op's window) |
| x head: first x block on the gate/up cores (med / max) | 16 / 23 | 16 / 23 | 15 / 22 | 16 / 24 | nothing; hidden under the weights' head + first expert | - |
| weight stream excess: (last weight landed max - first landed) - E x floor/expert | 49 | 41 | 93 | 100 | E (per-expert gaps 2.5-3.9 us + backpressure) and M (x / y reads interfere) | see the two rows below |
| ... of which per-expert switch gaps (median core) | 45 (21 base + 24 backpressure at experts 4-6) | 28 | ~100 | 28 | E | grant the next expert's slots before the current one drains (ring credit at the switch) |
| ... of which slot skew (last weight landed max - med) | 15 | 20 | 31 | 53 | E (accumulated imbalance across DRISC slots) | balance the slots' loads / work stealing at the end |
| tail: last weight landed -> last gate/up K loop | 59 | 24 | 17 | 16 | sub-blocks of the last expert (x passes ~13 us each at H 6144) | split the last expert's rows over idle cores / put the smallest expert last |
| tail: last gate/up -> last down / y write -> kernel end | 17 | 16 | 22 | 25 | nothing much (down of the last sub-block + y) | overlap with the next op (the fused flat + combine already does) |
| sum (head + stream excess + tail) | 135 | 91 | 146 | 152 | | |
| measured overhead above floor (profiled call) | 136 | 91 | 147 | 152 | | |
Takeaways: the head (10-14) and the drain (16-25) are true per-call constants; the rest scales. GLM-5.3's large tail
is its last expert's three 64-row sub-blocks re-running the x passes after its weights landed (59 us), and its stream
stalls behind the weight ring at experts 4-6 (backpressure: the ring holds two experts, the compute falls behind).
K3 / K2.7 pay mostly in the stream (per-expert switch gaps x 28, slot skew 31-53 us).
Logs / scripts: /localdev/mstaletovic/fixed_cost/ (fc_*.log the sweep, fit.py, pz_* / pz2_* + csv, pieces.py).

### Two attempts at each fixed cost (2026-10-26, one chip bh-lb-09, Galaxy loads; A/B = /localdev/mstaletovic/fc_ab/ab.sh)
Every attempt: GLM-5.3 E 8 M 160 / GLM-Flash E 9 M 142 / K3 E 28 M 91 / K2.7 E 12 M 107, us a call (median of 10),
plus a zones run (pieces.py) for the targeted piece, y at M 128 vs the base (bit-exact on every kept change). Base
525 / 382 / 1232 / 808. No hang in any run. Corrections to the fixed-cost table above, from the new marks: the head is
the launcher's job post (schedule load 1.4 us, then 4 us of descriptor building + one barrier for all 16 slots), and
the first math waits for x (first x block med 15.7, max 22.8 us), not the weights; the per-expert switch gap is
~0.5-1.3 us above a core's normal block spacing (2.0-2.6 us), the rest of the "gaps" was backpressure; K2.7's slot skew
is tile 1 (all job-1 slots, the NOC1-endpoint tiles, finish ~45 us after the job-0 slots: the x reads' port).
| fixed cost | attempt | GLM-5.3 | GLM-Flash | K3 | K2.7 | targeted piece | kept? | why |
|---|---|---|---|---|---|---|---|---|
| startup | 1a each slot's job count right behind its own descriptor (no barrier over all 16) | 523 | 382 | 1233 | 807 | first weight block -1.3..-1.7 us (med), -1.7..-3.1 (min) | yes | free; the start is x-bound (first x ~16 / 23 us), so the total does not move |
| startup | 1b 16-tile super-blocks (shorter x head, MIMO_FL_SBT=16) | 565 | 411 | (16 already) | 797 | first x block -1.5 us | no | the head shrinks but GLM-5.3 / Flash lose 30-40 us steady state (more, smaller x reads); K2.7 -11 |
| expert switch | 2a non-blocking grant refresh in the DRISC loop | 524 | 383 | 1234 | 806 | K2.7 gap 2.57 -> 2.36 us, others same | no | the refresh is rarely on the path (the gaps are backpressure, not the DRISC) |
| expert switch | 2b 3-expert weight ring (GU_NREG 3 + DRING 3, MT 2) | 539 | 392 | 1245 | 827 | backpressure gaps gone (GLM-5.3 13 / 35 -> 3 us) | no | +12-19 us everywhere: the bigger ring costs more than the stalls it removes |
| slot skew | 3a slots' cores by hops from the tile that really sends (switch mode: job 1 from the NOC1 endpoint), nearest pairs first | 522 | 381 | 1216 | 808 | GLM-5.3 skew 10 -> 4, K3 30 -> 27, K2.7 55 -> 55 | yes (default) | job 1 was planned from the free tile (10.7 real vs 6.8 believed hops, last slots 15-21); M 256-1024 unchanged / -1% |
| slot skew | 3b x reads on NOC0 (off tile 1's port), on top of 3a | 548 | 418 | 1295 | 820 | skew gone (K2.7 55 -> 15, K3 -> 6) | no | the skew was tile 1's x port, but x on NOC0 costs more than the skew everywhere |
| last-expert tail | 4a experts by sub-blocks, most first (uniform: no change) | 522 | 381 | 1234 | 806 | reversed zipf (biggest last) 560 / 412 / 1316 / 849 -> 527 / 398 / 1295 / 825 | yes (default) | the tail is the last expert's sub-blocks after its weights: put the smallest last |
| last-expert tail | 4b 32-row sub-blocks (MT 1) | 572 | 458 | 1239 | 821 | GLM-5.3 gate/up tail 56 -> 83 us | no | the tail is compute-bound per pass (~13 us a pass at 1 row tile, rp 1): more, smaller sub-blocks cost more |
| drain | 5a 16 h chains to the down cores (MIMO_FL_DCH=16) | 524 | 380 | 1251 | 805 | drain -2 us on the pair plans | no | within noise on GLM-5.3 / Flash / K2.7, K3 (units, 4 chains) +19 |
| drain | 5b 48 down cores (smaller final down block, MIMO_FL_ND=48) | 527 | 389 | 1242 | 807 | drain unchanged / +4 | no | the drain is the last h + one row tile of down, not the down width |
Kept together: 522 / 380 / 1216 / 807 (base 525 / 382 / 1232 / 808), y bit-exact at M 128 on all four, test_closeout_flat
9/9, test_closeout_plans, the fused flat + combine closeout (2x4) pass; 20k-call soaks of GLM-5.3 / K3 / K2.7 at their
Galaxy loads pass. (1a relies on the NoC rule the DRISC already uses: a write that has left the NIU is followed on the
same static VC and route by the atomic.) New probes: MIMO_FL_SBT (super-block width), MIMO_FL_DRISC_PLACE=0 (the old
round-robin). tt-npe could not model the placement: this install aborts on any workload, its own example included.
Logs: /localdev/mstaletovic/fc_ab/ (p1a..p5b, *z zones + csv, zbase / zr* zipf, final, soak_*).
