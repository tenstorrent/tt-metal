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
