# Grouped `unified_routed_expert_ffn`: Galaxy (32 x P150) validation results

Executes `GALAXY_VALIDATION_PLAN.md` on branch `zbaczewski/moe-ffn-grouped` at 5b21cb5e41b (on top of `main`
28238f903b3); the only code change made afterwards is the band-mode guard (next commit), re-verified on device. Host `bh-glx-120-b09u08`: 32 x tt-galaxy-bh (P150), TDP limit 130 W (high-power tier), AICLK
1350 MHz under load, **DDR at 14 Gbps** (tt-smi `DDR_SPEED` 0x36b0; nominal is 16 Gbps). Single-chip
worker grid 12x10 (WORKER dispatch takes one of the 13 columns), 8 DRAM channels.

The 14 Gbps DDR matters for reading the numbers: 8 channels at 14 Gbps is 448 GB/s, the same peak as the
7-channel 16 Gbps P100 the design was measured on. The "8/7 scaling" predicted in the plan for DRAM-bound
cases therefore cannot show up on this box; a 16 Gbps P150 galaxy (512 GB/s) should be re-measured.

## Step 0: build and smoke — PASS

Clean Release build from scratch. Smoke on one chip of the mesh (`test_grouped_routed_expert.py`):

| suite | result | min PCC |
|---|---|---|
| `x_rm and (G5r10 or G10r10 or G4r8) and bf4` (9 distributions x 3 geometries) | 27 passed, 0 failed (7 min) | 0.9799 |
| `cache_hit or count_clamp or all_empty or legacy_path` | 5 passed, 0 failed | 0.9799 |

Same 27 + 5 as on P100. The factory picked `grid=11x10` for every grouped case (`grid_cols` default 11,
so one of the 12 available columns idles, as expected).

## Step 1: single-chip A/B on a P150 — PASS, numbers match the P100 within 1-5%

DRAM read ceiling (`bench_dram_ceiling.py`, raw rows in `perf_data/dram_ceiling_p150.jsonl`): best 438 GB/s
(bank-direct 16 KB bursts) and 436 GB/s for all-core dual-NoC single-tile reads — 98% of the 448 GB/s
this board's DDR speed allows, and the same as the P100 (437 / 427). The plan's "expect ~460 GB/s"
assumed 16 Gbps DDR.

A/B (`bench_ab.py`, 7 distributions x bf4/bf8, median of 3, PCC checked in-run; raw rows in
`perf_data/ab_p150.jsonl`; `python make_report.py` regenerates the full tables from `results/`). Device time in us, speedup vs legacy:

| case | dtype | legacy | G5r10 | G10r10 | G4r8 | G5r10 m8 |
|---|---|---|---|---|---|---|
| kimi_u (12 x 107) | bf4 | 2360 | 999 (2.36x) | 991 (2.38x) | **961 (2.46x)** | 1030 (2.29x) |
| kimi_u | bf8 | 3442 | 1657 (2.08x) | **1582 (2.18x)** | 1636 (2.10x) | 1763 (1.95x) |
| kimi_zipf (640..32, 2 empty) | bf4 | 2172 | 1058 (2.05x) | 1276 (1.70x) | 1145 (1.90x) | **966 (2.25x)** |
| kimi_zipf | bf8 | 3126 | 1755 (1.78x) | 2236 (1.40x) | 1853 (1.69x) | **1683 (1.86x)** |
| kimi_e24 (24 x 107) | bf4 | 4762 | 1861 (2.56x) | **1805 (2.64x)** | 1906 (2.50x) | 1942 (2.45x) |
| kimi_e24 | bf8 | 6903 | 3152 (2.19x) | **3077 (2.24x)** | 3230 (2.14x) | 3345 (2.06x) |
| m3_u4 (4 x 160) | bf4 | 1110 | 493 (2.25x) | 741 (1.50x) | **475 (2.34x)** | 554 (2.00x) |
| m3_u4 | bf8 | 2430 | 698 (3.48x) | 1291 (1.88x) | 706 (3.44x) | **689 (3.52x)** |
| m3_u8 (8 x 160) | bf4 | 2210 | 932 (2.37x) | 1503 (1.47x) | **918 (2.41x)** | 1083 (2.04x) |
| m3_u8 | bf8 | 4849 | 1399 (3.47x) | 2609 (1.86x) | **1382 (3.51x)** | 1420 (3.41x) |
| m3_u16 (16 x 160) | bf4 | 4431 | 1909 (2.32x) | 3074 (1.44x) | **1804 (2.46x)** | 2187 (2.03x) |
| m3_u16 | bf8 | 9673 | 2855 (3.39x) | 5275 (1.83x) | **2725 (3.55x)** | 3006 (3.22x) |
| m3_skew8 (800..0) | bf4 | 2148 | 1123 (1.91x) | 1577 (1.36x) | 1163 (1.85x) | **1097 (1.96x)** |
| m3_skew8 | bf8 | 4917 | 1821 (2.70x) | 2706 (1.82x) | 1898 (2.59x) | **1722 (2.85x)** |

PCC min per config: 0.980 (Kimi bf4), 0.982 (M3 bf4), 0.999 (bf8) — identical to P100. Geometric-mean
speedup over these cases, P150 vs (P100):

| model | dtype | G5r10 | G10r10 | G4r8 | G5r10 m8 |
|---|---|---|---|---|---|
| Kimi | bf4 | **2.32** (2.29) | 2.20 (2.22) | 2.27 (2.20) | 2.33 (2.30) |
| Kimi | bf8 | **2.01** (1.92) | 1.90 (1.87) | 1.96 (1.83) | 1.96 (1.87) |
| M3 | bf4 | 2.21 (2.24) | 1.44 (1.44) | **2.25** (2.24) | 2.01 (2.07) |
| M3 | bf8 | 3.24 (3.18) | 1.85 (1.82) | **3.25** (3.17) | 3.24 (3.18) |

The P100 recommendation holds unchanged: `ffn_num_row_groups=5, ffn_grid_rows=10` as the default (within
~2% of the best on every case except many-small-uniform Kimi where G10r10 is 1-4% better), G4r8 as the
8-row fallback costs nothing here (it is even marginally best on M3 uniform cases).

### Band mode (`col_strided=1, grid_cols=8`)

**Fails the gate: intermittent hang. Refused at validation from this commit on** (`TT_FATAL(op.col_strided == 0)`
in `unified_routed_expert_ffn_device_operation.cpp`); the kernels and the `col_strided` argument stay in place for
the follow-up.

What was observed (all on one P150, kimi_u bf4, `perf_data/results` and `galaxy_logs/` on the host):

- `bench_ab.py --configs "G5r10;G5r10c8s1;G10r10c8s1"`: `G5r10c8s1` (R=2, grid 8x10) hung on its first
  dispatch after the non-band `G5r10` config had run in the same process; host spun 53 min until killed.
- Fresh process, single `G5r10c8s1` dispatch: passes, PCC 0.980 (with and without the full watcher).
- Replay of the bench sequence (`G5r10` x5, `G5r10c8s1` x5, `G10r10c8s1` x5): `G5r10c8s1` passes all 5
  (PCC 0.980), `G10r10c8s1` (R=1) hangs on dispatch 0 without the watcher and on dispatch 1 under the light
  watcher (`TT_METAL_WATCHER=2`, NoC sanitizer and asserts off). Reproduced twice out of two tries.
- Where it stops (light-watcher dump + ring traces, `tools/parse_ring_trace.py`): rows 3-9 of the 8x10 band grid
  have finished (`GW`); row 0 (group 0, second-round expert `item 10`) has its writers inside
  `noc_async_read_barrier` (`NRBW`) on the band-mode `up` read of K-block 0 and its readers waiting on
  `up_done`; row 2's writers sit inside the output DRAM write (`NAWW`, `W.drain`); row 1 waits on compute.
  No core is waiting on a semaphore that a live peer could still send: the stall is at the NoC/DRAM level, not in
  the act/valid protocol. Band mode is the only configuration that issues single reads of 36 KB (R=2) or 73 KB
  (R=1) per K-block (`(k_e-k_b) * N/8 tiles` contiguous from one bank, on both NoCs, from all 80 cores), so the
  first thing to try is issuing the run in <= 8 KB pieces so reads and the drain writes interleave, then re-run
  the sequence above.

Timing was never recorded (the sweep died before the first band-mode record), so the ">5% over G5r10" question
stays open; with 12 worker columns available and band mode limited to 8, the ceiling result above (bank-direct
16 KB bursts 438 vs all-core dual-NoC 436 GB/s) says there is no bandwidth left for it to win on this DDR speed.

### 12 columns (`grid_cols=12`, the full P150 worker width)

Same sweep with the twelfth worker column in use (raw rows `perf_data/ab_c12_p150.jsonl`; 13 is not available,
dispatch owns one of the 13 Tensix columns). PCC clean everywhere (min 0.9799). Device time in us, speedup vs legacy:

| case | dtype | legacy | G5r10 (11 col) | G5r10 c12 | G10r10 c12 | G4r8 c12 |
|---|---|---|---|---|---|---|
| kimi_u | bf4 | 2379 | 1008 (2.36x) | 1013 (2.35x) | 1088 (2.19x) | 992 (2.40x) |
| kimi_u | bf8 | 3449 | 1657 (2.08x) | 1663 (2.07x) | 1618 (2.13x) | 1644 (2.10x) |
| kimi_zipf | bf4 | 2171 | 1069 (2.03x) | 1081 (2.01x) | 1309 (1.66x) | 1181 (1.84x) |
| kimi_zipf | bf8 | 3105 | 1779 (1.75x) | 1769 (1.76x) | 2247 (1.38x) | 1884 (1.65x) |
| kimi_e24 | bf4 | 4741 | 1851 (2.56x) | 1879 (2.52x) | 1897 (2.50x) | 1986 (2.39x) |
| kimi_e24 | bf8 | 6907 | 3131 (2.21x) | 3193 (2.16x) | 3136 (2.20x) | 3288 (2.10x) |
| m3_u4 | bf4 | 1084 | 473 (2.29x) | **437 (2.48x)** | 734 (1.48x) | 436 (2.49x) |
| m3_u4 | bf8 | 2430 | 693 (3.50x) | **676 (3.60x)** | 1285 (1.89x) | 672 (3.62x) |
| m3_u8 | bf4 | 2208 | 925 (2.39x) | **866 (2.55x)** | 1458 (1.51x) | 876 (2.52x) |
| m3_u8 | bf8 | 4846 | 1402 (3.46x) | **1329 (3.65x)** | 2583 (1.88x) | 1350 (3.59x) |
| m3_u16 | bf4 | 4379 | 1890 (2.32x) | **1811 (2.42x)** | 2979 (1.47x) | 1737 (2.52x) |
| m3_u16 | bf8 | 9680 | 2849 (3.40x) | **2735 (3.54x)** | 5263 (1.84x) | 2665 (3.63x) |
| m3_skew8 | bf4 | 2137 | 1131 (1.89x) | **1080 (1.98x)** | 1550 (1.38x) | 1130 (1.89x) |
| m3_skew8 | bf8 | 4924 | 1863 (2.64x) | **1780 (2.77x)** | 2676 (1.84x) | 1815 (2.71x) |

Geomean G5r10 c12 / G5r10 11-col: Kimi 0.99 (bf4) / 0.99 (bf8), **M3 1.06 (bf4) / 1.04 (bf8)**. That is exactly
what the N padding predicts: the per-column tile count only drops when `ceil(N/12) < ceil(N/11)`. Kimi gate/up
stays at 6 tiles per column (64 tiles), only down goes 21 -> 19, and the extra column's share of the
multicast/handshake traffic eats that. M3 goes 9 -> 8 (gate/up, 96 tiles) and 18 -> 16 (down, 192 tiles) and
gains 4-6%. The L1 guard narrows the M3 gate/up K-block to 12 at 12 columns (logged `in0_block_w_gu=12`),
so part of the potential is left on the table.

Recommendation: `ffn_grid_cols=12` for MiniMax-M3 on P150 (and any model whose hidden/emb tile counts
divide better by 12 than 11); keep 11 for Kimi. It must stay 11 on P100 (11 worker columns).

## Step 2: does the model tolerate 10 rows?

- `TtMoe` creates its dispatch/shared-expert sub-devices only for the overlap phase and calls
  `clear_loaded_sub_device_manager()` before `self.routed_expert(...)` (`tt_moe.py`), so the expert FFN
  runs on the full worker grid with no sub-device restriction.
- Fabric routers live on ethernet cores, not Tensix; with WORKER dispatch the single-chip grid is 12x10 and
  the 8x4 mesh opened with `FABRIC_2D_TORUS_XY` reports the same: every grouped run in Step 3 logged
  `GROUPED: G=5 R=2 grid=11x10` (or `G=10 R=1 grid=11x10`) on all 32 chips and completed. **10 rows are usable
  in the model; the 8-row fallback is not needed.**

## Step 3: full-model gate on the 8x4 galaxy

`test_kimi_moe_perf.py -k galaxy` on the 8x4 torus-XY mesh (32 chips, realtime-profiler device time of one MoE
forward at the 5k chunk, warm pass; the grouped path selected with the env overrides, model code unchanged).
The gate asserts the LEGACY band, so the grouped runs fail the assertion by being faster; the number is the
logged total.

| model | config | device time | vs legacy | gate |
|---|---|---|---|---|
| Kimi-K2.7 (384 experts / top-8, 7168 emb) | legacy | 5.771 ms (24 programs) | — | PASS: in band [4.981, 5.847] around 5.414 ms (+8%, DDR-speed-widened) |
| Kimi-K2.7 | `TT_MOE_FFN_ROW_GROUPS=5 TT_MOE_FFN_GRID_ROWS=10` | **4.418 ms** | **-1.353 ms (-23.4%)** | below the legacy band (expected) |
| Kimi-K2.7 | `TT_MOE_FFN_ROW_GROUPS=10 TT_MOE_FFN_GRID_ROWS=10` | 4.583 ms | -1.188 ms (-20.6%) | below the legacy band |
| Kimi-K3 (896 experts / top-16, 3584 latent) | legacy | 8.527 ms (35 programs) | — | PASS: in band [7.868, 8.872] around 8.370 ms |
| Kimi-K3 | `TT_MOE_FFN_ROW_GROUPS=5 TT_MOE_FFN_GRID_ROWS=10` | **6.222 ms** | **-2.305 ms (-27.0%)** | below the legacy band |

The plan predicted the K2.7 layer at ~4.0-4.1 ms from a 5.41 ms baseline (expert FFN 2.4 -> 1.0-1.1 ms); the
measured saving of 1.35 ms is that prediction, on top of this box's 5.77 ms baseline (6.6% above the CI
baseline, consistent with 14 vs 16 Gbps DDR). The whole-layer gain equals the op-level gain, so the routing ops
around the FFN are not yet on the critical path. G5r10 beats G10r10 at the model level for K2.7 (12 experts
per chip; the single-chip A/B had them within 1%), so G5r10 stays the recommended default for both generations.

Correctness at model level: `test_ttnn_moe.py::test_kimi_moe[blackhole-kimi-torus-xy-8x4-kimi-5k-pcc]` with
`TT_MOE_FFN_ROW_GROUPS=5 TT_MOE_FFN_GRID_ROWS=10`: **PASSED, reference_output PCC 0.983558 (threshold 0.971)**,
7 min 41 s.

Both legacy baselines pass their gates unchanged, so the legacy path is unaffected by this branch on Galaxy.

### M3

`models/demos/minimax_m3/tests/unit/test_ep_moe_vs_ref.py` (real M3 dims, 128 experts / 4 per chip EP dispatch over
all 32 chips, **bf8 expert weights**, `single_bh_galaxy` descriptor, FABRIC_1D), legacy vs grouped, per-prompt PCC vs
the torch reference:

| prompt | legacy | `TT_MOE_FFN_ROW_GROUPS=5 TT_MOE_FFN_GRID_ROWS=10` |
|---|---|---|
| 0 | 0.96754 | 0.96758 |
| 1 | 0.96551 | 0.96553 |
| 2 | 0.96242 | 0.96243 |
| 3 | 0.97358 | 0.97367 |
| 4 | 0.96514 | 0.96517 |
| 5 | 0.96511 | 0.96514 |
| 6 | 0.96471 | 0.96474 |
| 7 | 0.97283 | 0.97290 |

Both PASSED; the grouped path tracks the legacy path to the 4th decimal on every prompt (grid `11x10` on all
chips). The op-level M3 bf8 speedups (3.4-3.5x uniform, 2.7x skewed) are in Step 1; there is no M3 MoE
device-perf gate in the repo to run.

M3 model-level perf was NOT measured: the only M3 whole-model perf test (`minimax_m3/tests/perf/test_model_perf.py::
test_model_fwd`, random weights, `PERF_LAYERS=4 PERF_EXPERTS=128 PERF_SEQ=2048`) dies with a host-side SIGFPE in
`tt/attention/prefill.py:attention_forward` on the LEGACY path (`TT_MOE_FFN_ROW_GROUPS=0`), i.e. before and
independent of the MoE; the grouped variant was not run. The real-weights M3 galaxy pipeline
(`galaxy_prefill_kv_pcc.py`, golden traces) is the place to take that number; the op-level M3 bf8 result above is
the expectation (expert FFN 3.4x faster on the uniform EP32 case).

## Decisions and follow-ups

- **Defaults stay LEGACY in this PR** (`ffn_num_row_groups=0` in `tt_moe.py` / `tt_minimax_moe.py`). Promoting
  `ffn_num_row_groups=5, ffn_grid_rows=10` passes every gate here, but it moves the Kimi device-perf gates by
  -23% / -27%, and their `expected_ns` must be re-cut on the CI galaxy (16 Gbps DDR), not on this 14 Gbps box.
  Predicted CI values from the measured ratios: K2.7 5,413,674 x 0.766 = ~4.14 ms, K3 8,369,824 x 0.730 =
  ~6.11 ms. Suggested as the next PR: flip the defaults and re-centre both gates from that run.
- Band mode: refused at validation (this commit); NoC-level stall documented above, fix direction given.
- `grid_cols=12` is validated (PCC clean) and worth 4-6% for M3 dims, nothing for Kimi (see "12 columns");
  `grid_cols=13` cannot be used on P150 because dispatch owns one Tensix column. Promote it together with the
  row-group default for M3.
- Re-measure the DRAM-bound cases (bf8) on a 16 Gbps P150 galaxy; expect up to 8/7 on those only.
- Legacy-reader race ports (report section "Races") are still open; both legacy gates passed here, so there is
  no evidence they fire on Galaxy under this workload.
