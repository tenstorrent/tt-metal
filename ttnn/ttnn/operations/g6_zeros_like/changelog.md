# Changelog: g6_zeros_like

## Phase 0 — Core Implementation
- **Date**: 2026-11-27
- **What was done**: initial implementation via the incremental pipeline (planner → implementer → verifier). The op is one writer kernel per participating core (BRISC/NoC1). It zero-fills one L1 page from the firmware zero region and writes it to every output page of its contiguous range. The page split is folded over the full grid. The input is never bound to the program.
- **SUPPORTED at Phase 0**: dtype = [float32]. Layout TILE, rank 4, tile-aligned last two dims and DRAM interleaved are structural `ValueError` checks, not axes. Both call forms (new output, `optional_tensor`) are supported.
- **Accuracy achieved**: bit-exact. PCC = 1.0, max_abs_err = 0, mean_abs_err = 0, abs RMS err = 0, max ULP = 0, 0 / 196608 elements ≠ +0.0. Relative RMS and the got/true ratio are undefined against an all-zero reference. Measured on the float32 production case in both call forms × 3 input draws (randn, random 32-bit patterns, NaN/inf/-0.0/subnormal) via `test_g6_zeros_like_precision_baseline.py`.
- **Golden suite at Phase 0**: 4 / 4 supported cells pass, 2 `xfail_expected` (bfloat8_b), 8 / 8 regression tests pass. `supported_fail = xpass_drift = xfail_wrong_mode = 0` (`verifier_report.json`).
- **Perf at Phase 0** (profiler device kernel time, median of 12): float32 (1,12,128,128) 3603 ns on 110 cores, against a goal of ≤ 2310 ns and stock 4920 ns. bfloat8_b (5,1,32,256), via direct dispatch, 819 ns on 40 cores.
- **Issues encountered** (verifier fixes):
  - The RT-arg reads and output-accessor setup ran after the zero-fill barrier and outside any profiler zone. Moved into the fill's shadow inside the `g6zl-zero-page` zone: kernel start → first write 253 → 243 ns; end-to-end unchanged within noise.
  - The `ZERO_SCRATCH_PAGES` depth knob did not reach the kernel, which always filled one page. It is now single-sourced through CT arg `zero_scratch_bytes`, which sizes both the CB and the fill.
  - `op_design.md` and `l1_ledger.md` were updated to match.
- **Tests added**: `test_g6_zeros_like_precision_baseline.py`, `test_g6_zeros_like_trace.py` (BB-16 trace capture and replay), and `test_profile_core_knobs_bf8` in `test_g6_zeros_like_perf.py`. Existing: `test_g6_zeros_like.py` (acceptance, 21 tests) and the `test_g6_zeros_like_perf.py` profiling driver.

## Refinement 1 — bfloat8_b
- Date: 2026-11-27
- What was done: added `ttnn.bfloat8_b` to `SUPPORTED["dtype"]`. No kernel or program-descriptor change: the page size already comes from `buffer_aligned_page_size()` (1088 B for bf8b), the CB format follows the output dtype, and all-zero bf8b bytes (shared exponents and mantissas) decode to +0. `test_profile` and `test_profile_core_knobs_bf8` now call the public entry point instead of dispatching the descriptor directly. No `compute_kernel_config`, tagger, EXCLUSIONS entry or placement retune, per the verifier notes.
- Accuracy achieved: bit-exact on bfloat8_b (5,1,32,256) new_output, randn input over a freed ones-filled block: PCC = 1.0, max_abs_err = 0, 0 of 40960 elements ≠ 0. Relative RMS is undefined against an all-zero reference. float32 unchanged (bit-exact).
- Golden test progress: 6 / 6 axis cells (the 2 bf8b cells, `test_op[5x1x32x256-new_output-dtype=BFLOAT8_B]` and its loose twin, moved from `xfail_expected` to pass). Ran the slice of both bf8b cells plus the float32 new_output pair. The float32 into_optional pair was not re-run; its path is unchanged.
- Perf-guard baseline (profiler device kernel time): bfloat8_b (5,1,32,256) new_output through the public entry point, 813.5 ns median of 16 samples (min 789, max 841), 40 cores. Matches Phase 0's 819 ns via direct dispatch.
- Issues encountered: None.
- Tests added: none. The existing `test_bfloat8_b` and `test_precision_bfloat8_b` now take their supported branch.

## Refinement 2 — DRAM write stream of the perf-flagged profile
- Date: 2026-11-27
- What was done: built the design's deferred `bank_run_coalesced` regime and made it the default for shapes where some run is ≥ 2 pages.
  - Kernel: a compile-time branch (CT arg `bank_run_stride`, 0 = page split) writes one in-bank run (page ids `b + B·k`, contiguous bytes in bank `b`) as `noc.async_write` chunks of `zero_scratch_bytes` from the one zero scratch, through the output `TensorAccessor`. Under `--dev` it ASSERTs that the host bank count is the accessor's `contiguous_page_stride()` and that the run is `num_contiguous_pages` long. The page-split path is unchanged.
  - Host: `REGIME` and `BANK_ASSIGNMENT` knobs (in `cache_key`). `_bank_run_split` splits each bank's pages over its cores. `segment_rows` places them so no two banks share a NoC link on the NOC_1 writer (north, then west). The bank→worker anchor map (`get_optimal_dram_bank_to_logical_worker_assignment`) gives segments (the worker block east of each DRAM column, which writes only that column's banks) and, inside them, row bands from each bank's endpoint row down to the next.
  - `NUM_CORES_CAP` changed `None` → 80, one rectangle `(0,0)-(7,9)`. `describe_plan` now reports `regime`, `page_stride` and `bank`.
  - Levers measured on float32 (1,12,128,128), median of 12 (device kernel ns):
    - Regime: page split 3587–3655 vs bank runs 2251–2290 at 110 cores.
    - Assignment at 110 cores: `bank_major` 2444–2532, `stagger` 3514, `nearest` 3159, `segment_rows` even chunks 2289–2301, `segment_rows` endpoint-row bands 2251–2270.
    - Core count, column-major: 70–90 cores flat at 2053–2121; 96–100 cores ~2180; 110 cores ~2260. Explicit column sets that skip the east segment's far columns (30–80 cores) all land at 2028–2102.
    - Transfer size: 8 KiB chunks (scratch 2) are 90–150 ns slower than 4 KiB. The bigger loopback fill on every core's critical path costs more than the saved commands, so `ZERO_SCRATCH_PAGES` stays 1 as a live knob.
    - Bank stagger: the `stagger` assignment was slower. With one bank per core, issue-order stagger does nothing.
    - Placement: row-major caps leave `segment_rows` bands empty and fall back. `ROW_WISE` stays `False`.
  - Result, same session as the pre-change code, 12 samples × 2 runs:
    - float32 new_output 3643 / 3522 → 2079 / 2091 ns.
    - float32 into_optional 3568 / 3618 → 2074 / 2096 ns.
    - bfloat8_b new_output 824 / 826 → 821 / 816 ns: unchanged, stays on the page split because P = C = 40.
  - Unprofiled trace replay (200 calls, float32 into a fixed buffer): 5.27 → 4.26 µs per call.
  - What binds now (zones at the shipped plan):
    - First→last core start ~450 ns, zero fill ~195 ns, write barrier median 0.92 µs / max 1.6 µs.
    - First write to last ack ≈ 1.8 µs for 786432 B ≈ 437 GB/s, near the ~460 GB/s DRAM roofline.
    - The rest is per-call fixed cost (launch skew + fill): Refinement 3.
- Accuracy achieved: bit-exact. 0 of 196608 float32 elements ≠ +0.0 in both call forms (precision baseline and acceptance; NaN-prefilled `optional_tensor`); bfloat8_b 0 of 40960. PCC = 1.0; relative RMS is undefined against an all-zero reference.
- Golden test progress: 14 / 14 (6 axis cells + 8 regression).
- Issues encountered:
  - The NoC estimator CLI (`/perf-ceiling-dm`) is not built in this checkout. The bracket was read straight from `noc_latencies.yaml`: BH DRAM-interleaved `ALL_TO_ALL` writes on NoC1 ≈ 2 B/cyc/core ≈ 300 GB/s at full grid.
  - `run_safe_pytest.sh --profile` splits a `-k` expression on spaces; explicit node IDs were used instead.
  - Profile `core_x/core_y` are physical coordinates: worker columns 1–6 and 11–15, harvested 7/10/16; DRAM at x=0 and x=9. `worker_core_from_logical_core` returns translated ones.
- Tests added:
  - `test_g6_zeros_like_perf.py`: `test_profile` now takes 12 samples and covers into_optional. New lamp drivers `test_profile_write_stream_fp32` (regime × assignment × cap × placement × scratch) and `test_profile_write_stream_bf8`, each writing a config→row map to `generated/g6_zeros_like_sweep_order.txt`.
  - `test_g6_zeros_like.py::test_plan_writes_every_page_once_on_many_cores` honors `page_stride`.

## Refinement 3 — Per-core fixed cost of the perf-flagged profile
- Date: 2026-11-27
- What was done: took the zero fill off the path to every core's first DRAM write.
  - Kernel: a compile-time `head_pages` (CT arg 4). The first `min(head_pages, num_pages)` pages of each core's range are written straight from the firmware `MEM_ZEROS` region. Each byte range gets one `set_async_write_state` and then one 512 B `async_write_with_state` per chunk; a ragged remainder (bfloat8_b's 64 B) is one plain write. The scratch fill runs only if pages remain, and the existing page-split / bank-run loops write the rest from the scratch. `head_pages = 0` keeps the old fill-first path with the setup in the fill's shadow. New zone `g6zl-zero-head`.
  - Host: knob `MEM_ZEROS_HEAD_PAGES = 2` (`None` = whole range), clamped to the longest range in the plan, in `cache_key`, and reported per core by `describe_plan` (`head_pages`).
  - Reused: both regimes, the scratch, the chunked bank-run and per-page write loops, the CB and its single-sourced size. Added: the head write, the knob, and a guard that skips the fill.
- Measurements (profiler device kernel time, median of 12):
  - Starting state, same session: float32 flagged shape 2116 ns.
  - Zones: every bank drains at 54–58 GB/s (~452 GB/s total) once its first write lands. The first write lands 244–328 ns after the first core starts, 195 ns of that being the fill.
  - Head sweep at 80 cores: 0 → 2064–2076, 1 → 1981–2002, 2 → 1929–1934, whole range → 1906–1920 ns. With the head, each bank's first write lands at 33–127 ns and the drain rate is unchanged.
  - Longer ranges, float32, 80 cores:
    - (1,48,128,128), 10 pages per core: 6760 / 6646 / 6687 ns for head 0 / 2 / whole.
    - (4,48,128,128), 38 pages per core: 28992 / 28614 / 33190 ns. Whole-range 512 B writes go issue-bound, so the shipped head is 2.
  - Shipped, `test_profile`: float32 new_output 1926 ns, into_optional 1940 ns; bfloat8_b new_output 691 ns, into_optional 690 ns (Refinement 1 baseline 813.5).
  - Core count with the head: caps 64 / 48 / 32 fall back to `bank_major` (east banks get no core) at 3.4–4.3 µs; 110 cores 2056–2248 ns. `NUM_CORES_CAP` stays 80.
  - No-payload ablation (same args, setup and barrier, no writes): 546 ns. Launch skew is 457 ns; per core ~88 ns.
  - Unprofiled trace replay (200 calls): 4.25–4.27 µs/call for head 0, head 2, whole range and the no-payload kernel alike. Replay is dispatch-bound, so it can show no regression but cannot resolve kernel deltas.
- What binds now: the per-bank DRAM drain, from ~50 ns after the first core starts. Bank 0 ends last in 8 of 12 profiled calls (~1.90 vs ~1.78 µs for banks 1 and 6). The profiler pushes each early-finishing core's buffer to DRAM bank `core_flat_id / cores_per_dram` (`kernel_profiler.hpp` `finish_profiler`), so low-id cores write into bank 0 during the drain: a profiler artifact. The remaining gap to the 1.71 µs roofline is ~200 ns (launch skew per bank plus that interference). The larger end-to-end term is the ~4.25 µs dispatch floor per traced call; it is host-side and outside this device-ns refinement.
- Accuracy achieved: bit-exact. float32: 0 of 196608 elements ≠ +0.0 in both call forms (acceptance, precision baseline, trace). bfloat8_b: 0 of 40960. Every head setting (0, 1, 2, whole) is exact on bank runs, bank runs with 2-page scratch, the float32 page split and bfloat8_b, plain and `--dev` (the NoC sanitizer accepts `MEM_ZEROS` as a write source). PCC = 1.0; relative RMS is undefined against an all-zero reference.
- Golden test progress: 14 / 14 (6 axis cells + 8 regression).
- Issues encountered:
  - The ablation kernel cannot call `Noc::write_zeros_dram_barrier()` without the zero-API include path that defines it (link error); it uses `async_write_barrier()`.
  - The sweep order file is written twice per test under `--profile` (up-front collect); dedupe it before mapping rows.
- Tests added:
  - `test_g6_zeros_like.py::test_mem_zeros_head_knob`: 16 cases, head 0 / 1 / 2 / whole × (float32 bank run, bank run with 2-page scratch, float32 page split, bfloat8_b) into a NaN / ones-prefilled `optional_tensor`.
  - `test_g6_zeros_like_perf.py`: lamp drivers `test_profile_fixed_cost_fp32` (head × cap), `test_profile_fixed_cost_bf8`, `test_profile_head_long_runs_fp32`.
  - `test_g6_zeros_like_ablation.py` with `probes/ablate_no_writes_writer.cpp`: the no-payload floor (perf only; output wrong by design).
