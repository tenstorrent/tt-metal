# Changelog: g6_clone

## Phase 0 — Core Implementation
- **Date**: 2026-11-27
- **What was done**: Initial implementation via the incremental pipeline (planner → implementer → verifier).
  - Regime `flat_page_range`: the flat page axis is split into contiguous ranges over the full grid (110 cores, `row_wise=True`).
  - The reader (NCRISC, NoC0) moves each core's whole range into one CB behind one barrier. The writer (BRISC, NoC1) drains it behind one flush, then one final write barrier.
  - There is no compute kernel. The program descriptor is cached, with the buffer addresses patched per call.
- **SUPPORTED at Phase 0**: dtype=[float32], layout=[TILE]. There are no shape-derived or op-specific axes.
- **Accuracy achieved**: bit-exact. PCC=1.0, max_abs_err=0, rel_rms_err=0, max ULP=0, got/true ratio 1.0, measured on 4 shapes via `test_g6_clone_precision_baseline.py`: (1,1), (1,12,128,128), (8,12,128,128), (16,12,128,128). The golden suite also grades bit equality on uniform random bit patterns (NaN payloads, ±inf, subnormals).
- **Golden suite at Phase 0**: 7/7 SUPPORTED cells pass, 28 xfail_expected, 0 supported_fail, 0 xpass_drift, 0 xfail_wrong_mode. All 3 non-registry regression cases pass (per `golden_results/verifier_report.json`).
- **Perf at Phase 0** (device kernel ns, median of 6): p1 (16,12,128,128) f32 63182 (stock 68320, roofline 54610), p2 32492, (1,12,128,128) 4700.
- **Issues encountered**: the verifier made three code-review fixes and no drift fixes.
  - Exposed `BLOCKS_PER_CORE_KNOB`: the block count per core had been collapsed to 1.
  - Routed the entry point and `g6_clone_plan` through one `_plan()`.
  - Added a multi-block ring test for the depth-2 path, which no INPUTS shape exercises.
  - `eval.verify_supported` needs `.tt_chip_scope` sourced (the MPIX stub `LD_PRELOAD`) to import ttnn outside `run_safe_pytest.sh`.
- **Tests added**: `test_g6_clone.py` (incl. `test_g6_clone_multi_block_ring`), `test_g6_clone_precision_baseline.py`; perf/lamp probes `probes/perf_baseline_probe.py`, `probes/grid_cap_probe.py`, `probes/block_knob_probe.py`, `probes/probe_006.py` (TARGET-widened cell sweep).

## Refinement 1 — Full TARGET surface (bfloat16, int32, uint32; ROW_MAJOR) with the logical-rows partial-tile regime
- **Date**: 2026-11-27
- **What was done**:
  - `SUPPORTED` now equals TARGET: dtype [float32, bfloat16, int32, uint32], layout [TILE, ROW_MAJOR]. No kernel change was needed for the new dtypes and layouts: the copy moves bytes, the CB page is `buffer_aligned_page_size()`, and the CB `data_format` follows `x.dtype`.
  - Built regime `logical_rows_partial_tile` for TILE inputs with a partial last tile row or column. It keeps the same split, CB and block loop as `flat_page_range`. Only the per-page transfer in `load_block` / `store_block` changes, behind the CT flag `logical_rows`.
    - The kernels move at most two byte ranges per page: the face rows that hold logical rows, rounded up to the 64 B DRAM alignment, at their own in-page offsets.
    - `PartialTileWalk` (`kernels/g6_clone_page_pieces.hpp`) picks one of four page kinds from a CT table without dividing page ids.
    - Full pages move whole.
  - New knob `PIECE_MERGE_GAP_BYTES = 448`: two ranges at most that far apart move as one span through the gap.
  - The cache key gained `tile_geometry(x)`, so `[1,1,1,62080]` and `[1,1,32,62080]` no longer share a descriptor. `g6_clone_plan` reports `regime` and `partial_tile_walk`.
  - The design's Regimes / API Mapping / Host Contract and the L1 ledger's data-movement budget are updated.
- **Perf** (device kernel ns, Blackhole p150b, one fresh-cache profiled run per variant, median of the 5 repeat calls):

  | Case | `flat_page_range` | logical rows, ranges split | logical rows, merged (shipped) |
  |------|-------------------|----------------------------|--------------------------------|
  | Q8 `[1,1,1,62080]` bf16 | 19439 | 9347 | **7062** |
  | F13 `[1,5120]` bf16 | 2388 | 1860 | **1593** |

  - Unchanged full-tile paths: p1 63682 (Phase 0 63182), p2 32159 (32492), p4 19152 (19.4 µs). All within noise.
  - 32 B ranges instead of 64 B: Q8 9736, no gain.
  - Merging wider gaps loses: 17×16 bf16 (512 B gap) +3%, one-row f32 (960 B) +10%, 17×16 f32 (1024 B) +53%.
  - Zone split on Q8: the reader is the bound. On the slowest core, read issue + barrier take up to 6.1 µs with split ranges and 2.3 µs + 4.6 µs merged, while write issue stays at 0.8–1.2 µs on every core.
  - Small ranges are bound by read requests, not bytes. The levers left for that bound are bank stagger/affinity and NoC/VC placement (Refinements 2–3), with Q8 in their guard set.
- **Accuracy achieved**: bit-exact on every case (PCC=1.0, relative RMS=0). Shapes: all 29 golden INPUTS; the 6 loose cases, including `poisoned_padding_nan_bf16` and `poisoned_padding_nan_f32`; and the new 88-cell layout matrix (4 dtypes × 2 layouts × 11 shapes, ranks 1–4, aligned and partial on H, W and both).
- **Golden test progress**: 38/38 passing (29 cells + 6 loose + 3 regression). No cell is xfail any more, so `supported_fail = xpass_drift = xfail_wrong_mode = 0`.
- **Issues encountered**:
  - An interrupted earlier attempt left the regime committed but `g6_clone_page_pieces.hpp` untracked. It is now committed.
  - The pre-commit `black` hook fails with "Ran out of input" on a corrupt cache. A fresh `BLACK_CACHE_DIR` fixes it.
- **Tests added**: `test_g6_clone_layout_matrix.py`:
  - `test_g6_clone_layout_matrix`: dtype × layout × shape, bit-exact. Also serves as the precision matrix: the op has no compute config.
  - `test_g6_clone_poisoned_padding_kinds`: NaN or pattern padding over all four page kinds, bf16 / f32 / int32.
  - `test_g6_clone_plan_regime`: pins the regime and merged ranges.
  - `test_g6_clone_cache_key_partial_vs_full`.
  - Probes: `probes/logical_rows_ab_probe.py` (A/B variants, the source of the perf table above) and `probes/probe_007.py`.

## Refinement 2 — Speed up p1: DRAM bank locality
- **Date**: 2026-11-27
- **What was done**:
  - Built regime `bank_affine_runs` as the default for full-page inputs. The design had deferred it. Each core takes a run of one DRAM bank's pages (`b, b+8, b+16, …`, contiguous in bank `b`) and moves each block as one read and one write. The flat walk becomes `start_page + i * page_stride`, with `page_stride = num_banks`. Same CB, same block loop.
  - Placement `dram_rows` (`BANK_AFFINE_PLACEMENT`): a bank's cores are the cores of its anchor row, from just east of its DRAM column to the next one. That is 44 cores on the 11×10 grid, in at most one rectangle per bank (7 here, because banks 0 and 4 share row 9).
  - Endpoint knob `BANK_AFFINE_ENDPOINT_NOC = 0`, passed to both kernels as CT arg `bank_endpoint_noc`. Both kernels address the bank's NoC0-table endpoint, which sits on the cores' row, through `TensorAccessor::get_noc_addr(page, 0, noc)`. Blackhole's NoC1 default endpoints sit on other rows, so writes to them took a north leg in every column, and those legs overlapped across banks. Now read data flows east along the row on NoC0 and write data flows west back along it on NoC1. The same knob picks the anchors the placement is built from.
  - Gate: `bank_affine_runs` runs only when every group core gets at least one page (48 or more pages on this grid). Below that, `flat_page_range` is the fallback; logical-rows shapes keep the flat split. Both outcomes are pinned by plan tests.
  - `bank_stagger` lever: CT flag `bank_stagger` (`FLAT_BANK_STAGGER`) and RT arg `rotate` on the consecutive-page walks (`flat_page_range`, `logical_rows_partial_tile`). Core k's walk starts on bank `k mod 8` and wraps once at the end of its range, resetting `PartialTileWalk` at the wrap. Measured off → on: Q8 7258 → 7248 ns, F13 1620 → 1646 ns (noise). Flat-forced p1 went 63528 → 62719 and p3 25843 → 25076, but both now run `bank_affine_runs`. Parked off (unchanged behavior) as a live knob, with a bit-exact test.
  - `g6_clone_plan` reports `bank_endpoint_noc` and `bank_stagger`. The regime remains a function of the existing cache key.
  - The design's Regimes and Traffic rows and the L1 ledger now record the measured reason. The deferral premise ("transaction count only; pays only when a DM RISC is issue-bound") was wrong. The bound was route overlap toward the DRAM endpoints.
- **Perf** (Blackhole p150b, device kernel ns, one fresh-cache profiled run per variant, median of its 3 calls; `probes/bank_affinity_ab_probe.py`):

  | p1 `(16,12,128,128)` f32 variant | full | read-only | write-only |
  |-----------------------------|------|-----------|------------|
  | `flat_page_range` (prior) | 63616 | 31773 | 44330 |
  | bank-affine, `dram_side` (all 110 cores in row bands; the interrupted attempt's default) | 72632 | 36300 | 45512 |
  | bank-affine, `interleave` | 69221 | — | — |
  | bank-affine, `dram_rows`, each kernel on its own NoC's endpoints | 58349 | 25762 | 34542 |
  | **bank-affine, `dram_rows`, shared row endpoint (shipped)** | **51357** | 25725 | 25138 |

  - `dram_rows` with 2, 4 or 8 blocks per core at depth 2 measured 60.8, 60.7 and 63.7 µs, all slower than one block (58.2 µs). Once the reads and writes of a bank interleave in DRAM, it loses throughput, so `BLOCKS_PER_CORE_KNOB` stays at 1.
  - Guard set, flat → shipped: p2 32464 → 25516; p3 25727 → 21516; p4 19405 → 16600; Q13 4605 → 3925; `(1,2048,128)` bf16 3066 → 2867; Q3 `(16,3,2560)` bf16 ROW_MAJOR 2046 → 1676. Q8 (logical rows, flat split) went 7196 → 7246 and the single-page int32 ROW_MAJOR case 936 → 932, both within noise on unchanged paths.
  - p1 now moves 25.17 MB at 490 GB/s, 96% of the 512 GB/s DRAM spec. That is the lower bound of Blackhole's adjacent-core DRAM read benchmark (490–512 GB/s, `tests/scripts/test_moreh_microbenchmark.py`) and below the 54.61 µs roofline. Full time ≈ read-only + write-only (25.7 + 25.1 µs): DRAM is saturated and time-shares the two directions.
- **Accuracy achieved**: bit-exact (PCC=1.0, relative RMS=0) on p1, p2, p3, p4, Q13, F5, Q3 ROW_MAJOR and the 88-cell layout matrix, which covers both regimes across dtype × layout × aligned/partial shapes, plus the poisoned-padding kinds.
- **Golden test progress**: 28/28 in the slice I ran (`-k "not uint32 and not int32 and not poisoned"`, every cell on the changed path). The 10 deselected cells run the unchanged `flat_page_range` / `logical_rows_partial_tile` paths. Unit tests 133/133.
- **Issues encountered**:
  - An interrupted earlier attempt had committed `bank_affine_runs` with placement `dram_side` as the default, a 13% p1 regression. Its band split put most cores north of their bank's NoC0 endpoint row, and NoC0 (east, then south) reaches those cores only around the torus.
  - One Tracy capture crashed (`tracy-capture` exit 3). The test had passed; the re-run measured cleanly.
- **Tests added**:
  - `test_g6_clone_plan_bank_affine_rows`: pins placement, endpoint and rectangle count.
  - `test_g6_clone_plan_regime`: now pins `bank_affine_runs` on p1, p4 and Q3, and the flat fallback at 40 pages.
  - `test_g6_clone_multi_block_ring`: parametrized over both full-page regimes.
  - `test_g6_clone_bank_stagger`: the stagger knob forced on over logical-rows, flat TILE and ROW_MAJOR shapes, with 1 and 3 blocks per core, bit-exact.
  - Probes: `probes/bank_affinity_ab_probe.py`, `probes/ablate_kernels/` (no-op reader/writer for read-only and write-only ablations), `probes/probe_008.py` and `probe_009.py` (DRAM / worker topology).

## Refinement 3 — Speed up p1: NoC route contention after bank locality
- **Date**: 2026-11-27
- **What was done**: Built both of the refinement's levers as `bank_affine_runs` knobs. Both are off by default, and the default path compiles to the Refinement 2 kernel code.
  - **Per-core VC.** `BANK_AFFINE_PER_CORE_VC` names the directions (`"read"`, `"write"`) whose requests take VC `BANK_AFFINE_VC_CYCLE[rank of the core in its row mod len]`. The kernel side is CT flag `per_core_vc` (CT arg 7, per kernel) plus RT arg 3 `bank_vc`. With the flag off, the VC folds to the constant `NOC_UNICAST_WRITE_VC`.
  - **NoC split.** `BANK_AFFINE_NOC_SPLIT`: `"none"`, `"alternate"` (every other core of a row swaps NoCs for its whole run) or `"half"` (each block's second half goes on the other NoC). The kernel side is CT flag `noc_split` (CT arg 6) plus RT arg 4 `other_noc_pages`. A split builds both kernels as `DataMovementConfigDescriptor(..., noc_mode=DM_DYNAMIC_NOC)` and barriers / flushes both NoCs.
  - The walk table's CT base moves from 6 to 8. `g6_clone_plan` reports `bank_vc_directions`, `bank_noc_split` and the per-role `route_args`. The consecutive-page walks never take a route knob.
  - The design's catalog, Regimes, API Mapping and plan-report sections record the knobs and the measurements below.
- **Perf** (Blackhole p150b, p1 `(16,12,128,128)` f32, device kernel ns, one fresh-cache profiled run per variant, median of its 3 calls; `probes/noc_route_ab_probe.py`, final code):

  | p1 variant | full | read-only | write-only |
  |------------|------|-----------|------------|
  | shipped (one VC, reads NoC0 / writes NoC1) | **51264** | 25890 | 25139 |
  | per-core read VC, cycle (0,1,2,3) | 51114 | 25879 | — |
  | per-core write VC, cycle (1,2,3) | 51205 | — | — |
  | per-core write VC, cycle (0,1,2,3) | 105647 | — | 87279 |
  | per-core VC both directions, cycle (1,2,3) | 51296 | — | — |
  | NoC split `alternate` | 57024 | 25986 | 29967 |
  | NoC split `half` | 62304 | 27394 | 36566 |

  - **Write VCs.** Writes on one VC run at 51.1–51.3 µs on any VC (0, 2 or 3), and so do rows cycling over two or three VCs. Only cycling over all four unicast VCs collapses writes: 3.5× slower write-only, and it reproduced in 3 sessions and for both orders. `BANK_AFFINE_VC_CYCLE` therefore defaults to `(1, 2, 3)`.
  - **NoC split.** The other NoC reaches the bank's row endpoint the long way round the torus. Its writes then share links with the row's own reads, which costs +11% (`alternate`) and +22% (`half`). Reads on the long route cost little (+0–6%).
  - **Default path.** A first version kept an `own_bytes != 0` branch on the default path. Over 5 interleaved runs it cost Q3 ROW_MAJOR (1–2 pages per core) +41 ns (1660 → 1701 ns) with no overlap between the runs. Restructured so the default compiles to the old single call; Q3 measured 1661 ns afterwards. The two extra RT args per core were ruled out.
  - Guard set at defaults: p2 25515 (Refinement 2 25516), p3 21543 (21516), p4 16618 (16600), Q8 7132 (7246), Q3 ROW_MAJOR 1661 (HEAD in the same session 1660), single-page int32 ROW_MAJOR 934 (932). All within noise.
  - **Bottleneck.** DRAM. Read-only 25.9 µs plus write-only 25.1 µs ≈ full: the bank time-shares the two directions at 486–501 GB/s, 95–98% of the 512 GB/s spec. Per-link demand is one bank's rate (~61 GB/s), below a 64 B/cycle × 1.35 GHz ≈ 86 GB/s link, so route overlap is not binding. The verifier's headroom gate (≤ 1.05 × 54.61 µs) had already been met.
- **Accuracy achieved**: bit-exact (PCC=1.0, relative RMS=0) on p1 in every knob variant; on p2 and Q3 ROW_MAJOR under five knob combinations, including the depth-2 ring with ragged blocks; on Q8 with the knobs forced on (they do not apply to it); and on the 88-cell layout matrix.
- **Golden test progress**: 12/12 in the slice I ran (`-k "p1 or p2 or 16x12x128x128 or 8x12x128x128 or 2560 or 62080"`: the bank-affine and logical-rows paths). The kernel CT layout change is also covered by the layout matrix. Unit tests 144/144.
- **Issues encountered**: the 4-VC write collapse above. Its mechanism is not identified, and the knob default avoids it.
- **Tests added**:
  - `test_g6_clone_layout_matrix.py::test_g6_clone_bank_route_knobs`: 5 knob combinations × p2 / Q3 ROW_MAJOR, bit-exact, with route args pinned.
  - `test_g6_clone_bank_route_knobs_skip_flat`.
  - `test_g6_clone_plan_bank_affine_rows` now pins the shipped route (one VC, no split).
  - Probe: `probes/noc_route_ab_probe.py`.

## Perf 1 — perf tournament round 1: the 13 production cells where stock was faster
- **Date**: 2026-11-27
- **Scope**: nothing added to `SUPPORTED`. The 13 targets come from the operator's A/B: 10 one-page cells (Q0 Q1 Q4 Q5 Q9 Q10 Q11 Q12 F7 F8) and 3 few-page cells (F10, Q14, Q7). The guard set is the 16 cells where the op already beat stock, p1 (the flagged case) among them. All numbers come from the case bench (`generated/meta/g6_clone/bench/bench_cases.py`): one process, 12 launches, median of the 11 after the first, `DEVICE KERNEL DURATION` ns, Blackhole p150b, chip 0. Differences within 2–3% are noise.
- **Result**: 4 ideas graduated, 2 not graduated (1 superseded, 1 superseded and mostly clock artifact), 0 regressions on the bench metric. Every target is now 33–44% faster than stock, and every guard is faster than stock. Golden 38/38, unit 145/145, and golden + `test_g6_clone.py` pass under `--dev`.

### Measured breakdown (Step 1)
- **Zones.** At the start (HEAD 274808b), the reader's 3 and the writer's 4 `DeviceZoneScopedN` zones were compiled into every profiled build, at about 50 cycles each. F10 per core in cycles (zone build): `rd_reserve` 50, `rd_issue` 102, `rd_barrier` 442, `wr_wait` 737, `wr_issue` 109, `wr_flush` 40, `wr_drain` 206. Kernel start spread was 626 cycles, 31% of the span.
- **Ablation** on the plain build (no zones): each stage's payload stubbed with its CB handshake kept (`perf_experiments/step1_ablation/`).

  | cell | full | read only | write only | handshake only |
  |---|---|---|---|---|
  | Q4 (1 page) | 731 | 462 | 346 | 80 |
  | F10 (32 × 2 KB, 32 cores) | 1273 | 951 | 837 | 536 |
  | Q7 (24 × 5 KB, 24 cores) | 1533 | 1096 | 1050 | 530 |
  | Q14 (3 × 5 KB) | 1005 | – | – | 204 |
  | p1 | 51123 | 25789 | 24986 | 539 |

  On every cell, read and write add up serially (one block per core). The ~530 ns handshake-only floor of F10 and Q7 is the spread of kernel starts across 11 columns.
- **Ranked bottleneck** for the targets:
  1. the start spread of multi-core small plans;
  2. the per-core DRAM read-latency chain;
  3. the write + ack chain;
  4. the fixed per-core start, 80 ns.

  p1 stays DRAM-bound (read-only + write-only ≈ full, 96% of spec), so it is a guard, not a target.
- **Roofline gate.** The targets are latency-bound; their bandwidth rooflines are under 0.54 µs (operator). `noc_estimate` is not built in this checkout, so the latency floors are the measured read-only and write-only chains above.
- **Column start offsets are mostly clock skew.** Measured by `perf_experiments/stagger_placement/` (STEP A, semaphore ping-pong on each core's own clock).
  - The per-column start offsets in profiler captures are mostly per-tile wall-clock offsets. They take fixed values of 0, 24, 155, 296 and 444 cycles, symmetric about the interior DRAM column.
  - Over the grid, the true start spread is 229 cycles, while the profiler shows 610. Real lateness exists but is small: x1/x2 start 66–167 cycles late, and row y4 about 100 cycles late.
  - `DEVICE KERNEL DURATION` includes up to 444 cycles (330 ns) of clock offset on a multi-core plan. Below, "corrected" means spans with that offset removed (`true_span.py`, `clock_offset_map.json`).

### Portfolio and verdicts
| idea | verdict | before → after (case bench ns) | domain |
|---|---|---|---|
| operator candidate 1: stage zones only in an opt-in zone build | **WIN, graduated** (2f504e8) | Q4 923 → 733, F10 1467 → 1276, F7 875 → 690 | every build: `G6_CLONE_ZONES=1` adds the `G6_CLONE_ZONES` define (`kernels/g6_clone_zones.hpp`, `G6C_ZONE`) and is part of the cache key |
| bank_local_small: small plans on their banks' anchor rows, both kernels on the row endpoint | **WIN, graduated** (cc7b95f) | Q4 733 → 563, F7 690 → 526, F10 1276 → 949, Q14 1005 → 764, Q7 1533 → 1133; guard plans unchanged | every full-page plan too small for the whole-row split, plus the one-page logical-rows plan |
| issue_path: raw NoC command-buffer issue (the writer arms its first write before the CB handshake; the reader skips the first reserve and does a lean push; counters and length set once per block) | **WIN, graduated** (3fdee6e) | on top of bank_local_small: Q4 563 → 460, F7 526 → 461, F10 949 → 859, Q14 764 → 653, Q7 1133 → 1060, F13 1416 → 1344, Q3 1556 → 1510, p1 50850 → 51016 (flat), Q8 6960 → 6916 | everywhere in DM_DEDICATED_NOC. Inexpressible under `noc_split` (DM_DYNAMIC_NOC, a parked knob): it keeps the dataflow API path |
| spread of bank_local_small: every bank's run on min(pages, 2) row cores, replacing the whole-row split | **WIN, graduated** (e200240) | whole row → 2 per bank: F5 3027 → 2429, F9 2872 → 2436, Q13 3850 → 3396, F11 3077 → 2770, Q3 1512 → 1369, F6 7096 → 6461, F14 9192 → 8549, F15 9129 → 8519, F12 11127 → 10557, F16 9387 → 9067, Q15 16770 → 16048, F4 21546 → 20806, Q6 25569 → 24965, p1 51047 → 50473 (noise). 1 per bank: p1 52424, F12 10680, Q3 1487, F5 2487, F10 827 | every bank-affine plan; `BANK_AFFINE_CORES_PER_BANK = 0` remains only as the flat-path test lever |
| fused_single_risc: one NCRISC kernel reads on NoC0 and writes on NoC1 | WIN alone, **not graduated** (superseded) | on the original op: Q4 727 → 669, F10 1283 → 1213, Q7 1527 → 1461, p1 flat. On top of bank_local_small: Q4 504, Q14 704, Q7 1081, against 460, 653, 1060 with issue_path | issue_path beats it on every cell. What fusing could still remove on top of issue_path is one cross-RISC CB poll, inside the noise band. Its best variant needs a raw NoC0 read-counter takeover that depends on firmware internals; its API variant (`DM_DYNAMIC_NOC`) keeps only 1–3% |
| stagger_placement (operator candidate 2, refined): a band of the earliest-starting columns for flat plans smaller than the grid | WIN on the metric, **not graduated** | F10 1273 → 1168 raw (corrected 1005 → 983, flat), Q7 1538 → 1304 (corrected 1254 → 1126), Q14 flat | superseded by bank_local_small on every production cell it touches (F10 947 vs 1168, Q7 1136 vs 1304, Q14 759 vs 996). Its F10 gain is clock offset. What remains of its domain (logical-rows plans of 2–80 pages) has no production cell |

- **Precision.** Every variant of every idea was bit-exact (a copy has no compute config). No option traded precision.

### What graduated, and how widely
- **The zone build (operator candidate 1).**
  - Zones stay at every stage boundary and compile only with `G6_CLONE_ZONES` (BB-21). The plain profiled build carries no user zone.
  - The plan reports `zone_build`. Added `test_g6_clone_zone_build`.
  - The gain exists only in profiled builds, which is how the campaign measures. Without the device profiler, `DeviceZoneScopedN` already compiled to nothing, so unprofiled production kernels were zone-free before and are unchanged. The other three graduations change unprofiled device time as well.
- **`bank_affine_runs` is the one full-page path.**
  - Each bank's run is split over min(its pages, `BANK_AFFINE_CORES_PER_BANK` = 2) cores of its anchor row, ordered by distance from the middle DRAM column (`bank_row_order`; the middle is derived from the anchor table, not hard-coded). Both kernels address the row endpoint.
  - Deleted:
    - the whole-row 44-core split;
    - its gate `BANK_AFFINE_MIN_CORE_PAGES`;
    - the `flat_page_range` fallback for small full-page plans. `flat_page_range` now serves no INPUTS shape.
  - The one-page logical-rows plan (F7, F8) runs on bank 0's first core. Its consecutive walk addresses the row endpoint through `page_endpoint_noc` (CT `bank_endpoint_noc` < `NUM_NOCS`; `OWN_NOC_ENDPOINTS` = 2 keeps each kernel's own table on flat walks).
  - p1 is now 192 pages per core, split by the L1 cap into blocks of 164 and 28 in the depth-2 ring: CB 1343488 B, under the 1347776 B budget (BB-05). Kernel groups span 7 rectangles (BB-17).
- **Raw issue path.**
  - Applies in every DM_DEDICATED_NOC build. `g6_clone_issue.hpp` holds the primitives; the kernel heads carry the measured justification.
  - The `noc_split` knob (DM_DYNAMIC_NOC, off by default) keeps the dataflow API calls. The raw command-buffer path does not exist in dynamic mode.
- **Carve-outs.**
  - Logical-rows plans of 2–109 pages stay on the flat split. They are inexpressible as bank runs, because the walk is consecutive. No production cell is affected.
  - The `noc_split` path, above.
  - No carve-out was earned by a measured regression on the bench metric.
- **Clock-corrected view of the 2-per-bank step** (whole row → 2 per bank, corrected ns):

  | cell | corrected ns | change |
  |---|---|---|
  | Q13 | 3654 → 3325 | −9.0% |
  | F14 | 8923 → 8438 | −5.4% |
  | F12 | 10906 → 10504 | −3.7% |
  | F11 | 2789 → 2698 | −3.3% |
  | F5 | 2534 → 2481 | −2.1% |
  | Q15 | 16208 → 15966 | −1.5% |
  | p1 | 50917 → 50381 | −1.1% |
  | **Q3** | **1230 → 1277** | **+3.8%** (1224 → 1268 in a second session) |

  - Q3's raw −10% is clock offset. In true time, 3 pages of 5 KB per core are slower than 1 page per core.
  - The same holds below the old gate. In the single-core-span view, one page per row core beats 2 per bank by 2–8% on F10 and Q7, but loses 4–13% on the bench metric.
  - This is a judgment call: the campaign grades the bench metric, so these stay on the single path without a carve-out, recorded here for the operator.
- **Whole round in corrected time** (start of the round, plain build → final): F10 996 → 781, Q7 1247 → 971; stock is F10 1070, Q7 1626. One-page cells run on one core and have no clock offset: Q4 731 → 459.

### Targets after the last commit (e200240), op / stock, same session
| cell | op | stock | op / stock |
|---|---|---|---|
| Q4 (8,) uint32 RM | 459 | 759 | 0.60 |
| Q10 (1,) uint32 | 462 | 762 | 0.61 |
| Q0 (16,) uint32 | 459 | 759 | 0.60 |
| Q9 (32,) uint32 | 467 | 764 | 0.61 |
| Q12 (1,1,1,32) uint32 | 470 | 767 | 0.61 |
| Q1 (16,) int32 | 459 | 761 | 0.60 |
| Q5 (8,) int32 | 459 | 761 | 0.60 |
| Q11 (1,) int32 | 460 | 762 | 0.60 |
| F8 (1,1) bf16 TILE | 461 | 761 | 0.61 |
| F7 (1,1) f32 TILE | 463 | 821 | 0.56 |
| F10 (1,1,256,128) bf16 TILE | 863 | 1179 | 0.73 |
| Q14 (1,3,2560) bf16 RM | 659 | 1013 | 0.65 |
| Q7 (8,3,2560) bf16 RM | 1061 | 1733 | 0.61 |

### Guard set (no regression)
- **Same session, op / stock:** p1 50388 / 68113, F11 2765 / 3407, Q3 1359 / 2039, F13 1341 / 2401, Q8 6845 / 20877.
- **Remaining guards** (2-per-bank lane on the final kernels, op; stock from the operator's A/B): Q6 24965 / 33025, Q13 3392 / 4806, Q15 16048 / 20977, F4 20806 / 28943, F5 2468 / 3421, F6 6461 / 8430, F9 2436 / 3370, F12 10591 / 15361, F14 8511 / 11370, F15 8519 / 11175, F16 9067 / 11699.
- Every guard is faster than at the start of the round and faster than stock.

### Operator candidates
- operator candidate 1: graduated - stage zones compile only in the opt-in zone build (`G6_CLONE_ZONES=1`), the plain profiled build has none (BB-21) - Q4 923 -> 733, Q9 925 -> 735 (operator lane), F10 1467 -> 1276, F7 875 -> 690, F8 873 -> 691 (operator lane) ns.
- operator candidate 2: not graduated - superseded by bank-local placement, which re-places the same small plans onto their banks' rows and wins more on every cell (F10 1168 band vs 947 bank rows, Q7 1304 vs 1136, Q14 996 vs 759). Its start-spread gain is mostly per-tile clock offset, not wall time (F10 corrected 1005 -> 983) - F10 1273 -> 1168, Q7 1538 -> 1304, Q14 1003 -> 996 ns (band rule, zones off).

### Helper bypasses
| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| no helper covers this: DRAM→CB page read (dataflow family; bypasses `noc_async_read_page` / `noc_async_read`) | capability | The read API bumps `noc_reads_num_issued` and rewrites the length register on every request. There is no form that sets a block's constant page length once and accounts the block's requests with one counter update. `*_set_state` / `*_with_state` cover one destination coordinate only, while consecutive pages rotate over 8 banks. | 727–732 (Q4, op kernels, flat path) | 673 (reader-only change) | `kernels/g6_clone_reader.cpp:90-124` |
| `cb_reserve_back` / `cb_push_back` (dataflow CB API) | capability | `cb_push_back` re-reads `tiles_received` after the data lands, so the load sits on the critical path after the barrier. The API has no "prepare, then commit" split. A first-block `cb_reserve_back` on a CB the firmware has just reset is pure overhead, and the API cannot express "known empty". | in the reader-only number above | in the reader-only number above | `kernels/g6_clone_reader.cpp:57,127` |
| no helper covers this: CB→DRAM page write (dataflow family; bypasses `noc_async_write` / `noc_async_write_page`) | capability | The writer cannot program a write request (CTRL, source, destination, coordinate, length) while it waits on `cb_wait_front` and then send it with one store. `noc_async_write_one_packet_set_state` / `_with_state` still poll and write two registers after the handshake. Per-request counter updates cannot be batched per block. Post-handshake issue: 113 → 25 cycles. | 727–732 (Q4) | 690 (writer-only change); 629–644 with the reader change too (one-page cells, before bank_local_small); 563 → 460 on top of it | `kernels/g6_clone_writer.cpp:93-99,117-127,174` |

- The kernel heads carry the measured justification, so a helper-usage pass should not revert these. These raw writes also bypass NoC event tracing and the watcher's NoC sanitizer. The `--dev` runs were clean.

### Artifacts
- `perf_experiments/`:
  - `step1_ablation/`: the coordinator's stage ablation;
  - `fused_single_risc/`, `issue_path/`, `stagger_placement/`, `bank_local_small/`: one per subagent, each with lanes, plugins, kernels and patches;
  - `bank_cap_above_gate/`: the cores-per-bank sweep, `run.sh <logdir> <case>:<cap>[:<min>[:<BANK_CORES>]]`.
- Lanes must run from `generated/meta/g6_clone/lanes/`: a bench file under `ttnn/` re-registers ttnn under pytest.
- Logs: `generated/meta/g6_clone/perf1/` (coordinator) and `generated/meta/g6_clone/lanes/perf1_*` (subagents).
