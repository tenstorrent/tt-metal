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
