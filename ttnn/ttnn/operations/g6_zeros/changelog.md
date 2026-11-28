# Changelog: g6_zeros

## Phase 0 — Core Implementation
- **Date**: 2026-11-27
- **What was done**: initial implementation via the incremental pipeline (planner → implementer → verifier). Regime `bank_run`: one writer kernel (BRISC, NoC1). 104 writer cores (13 per DRAM bank), each streaming one contiguous same-bank slot run from a 16 KiB core-local zero source, with one write barrier.
- **SUPPORTED at Phase 0**: dtype=[float32], layout=[TILE]. No op-specific axes.
- **Accuracy achieved**: exact. PCC=1.0, max_abs_err=0.0, rms_err=0.0, max ULP 0, no -0.0 (measured on 2 supported shapes via test_g6_zeros_precision_baseline.py). bf16 p4 and bf8b p1 are also exact past `validate()` (probe_029, `--dev`).
- **Golden suite at Phase 0**: 13 / 19 cells passing per `verifier_report.json`. The other 6 are `xfail_expected` (bf16 / bf8b → Refinement 1). 0 supported_fail, 0 xpass_drift, 0 xfail_wrong_mode.
- **Device time**: p1 bf8b 1,102,674 ns (259 GB/s; target ≤ 837,190 ns → Refinement 2), p2 48,693 ns, p3 24,827 ns, p4 1,603 ns.
- **Issues encountered (verifier fixes)**:
  - one plan derivation shared by `describe_plan` and dispatch, cross-checked against the buffer;
  - knobs resolved at call time and included in the host plan-cache key;
  - all kernel work moved inside named profiler zones (BB-21);
  - a cache-hit address re-patch test added;
  - four stale debug/perf test files that targeted superseded APIs removed;
  - the precision baseline's host memory bounded.
- **Tests added**: test_g6_zeros.py (acceptance, + cache-hit address case), test_g6_zeros_precision_baseline.py, test_g6_zeros_perf.py (knob sweep harness).

### Implementer build log (Phase 0)

Three builds, in order. The last one, "Rebuild from the bank-major zero-run design", is the current code. The first two were superseded, and their knob names (`Knobs`, `CORE_ORDER`, `WRITER_CORES_PER_BANK`) no longer exist. Their measurements agree with the verifier's re-measurement on the current kernel: bank-affine, one writer per bank, gives about 559.6 µs on p1.

### Phase 0 (implementer)

Built the design's single regime (bank-run zero write): one writer kernel (RISCV_0, NoC1), one
`cb_zero_source` of `Z` bytes zeroed once per core by NoC loopback, each core's share of one DRAM
bank written as single-packet writes from that source, one write barrier. No reader or compute
kernel (the design's kernel set). SUPPORTED: `dtype` float32, `layout` TILE.

#### Deviation from the design's Phase 0 knob values (measured)

The design's Phase 0 values were `num_worker_cores` = full grid (110) with round-robin `c mod B`
placement and `Z = min(burst, largest run)`. It listed core count, placement and `Z` as perf lamps
to measure. Measured on Blackhole (8 DRAM banks, 11x10 grid); the defaults now take the winners.
All three stay live knobs (`Knobs` in `g6_zeros_program_descriptor.py`).

Device kernel duration, ns (one fresh run each; the p1 and p4 rows that decided `Z` were repeated
3x, spread < 1%):

| shape | W=8 dram_adjacent Z<=4K (default) | W=8 dram_adjacent Z<=16K | W=8 row_major | W=16 row_major | W=110 row_major (design Phase 0) |
|---|---|---|---|---|---|
| p1 (32831,1,32,256) bfp8, 285.8 MB | 559,491 (510.8 GB/s) | 559,603 | 902,311 | 959,613 | 1,106,365 |
| p2 (16,12,128,128) fp32, 12.6 MB | 25,384 | 25,693 | 40,706 | 43,405 | 47,645 |
| p3 (8,12,128,128) fp32, 6.3 MB | 13,099 | 13,421 | 20,830 | 22,319 | 24,129 |
| p4 (1,4,256,128) bf16, 0.26 MB | 1,425 | 1,721 | 1,853 | 2,165 | 1,707 |

Unprofiled host time per call (steady repetition, one sync; RB-053) agrees: p1 563 us at W=8
dram_adjacent vs 1112 us at W=110; p2 26.2 vs 49.2 us.

- Bound: the DRAM banks. p1 runs at 510.8 GB/s, at Blackhole's nominal 512 GB/s DRAM peak, so no
  data-movement headroom is left on the DRAM-bound shapes.
- More writers per bank is slower (W=8 < 16 < 110 row-major). Placement matters most: the writer
  nearest each bank on NoC1 (`device.get_optimal_dram_bank_to_logical_worker_assignment(NOC_1)`)
  is 1.6x faster than the same 8 writers in row 0, where their NoC paths cross.
- `Z` does not move the DRAM-bound shapes between 4096 and 16384. It is fixed cost on p4 (the
  zeroing runs before the first write), so the default caps it at 4096.
- The `dram_adjacent` group is 8 single-core ranges (two columns, 4 rows each), not one rectangle
  (BB-17). The 8 ranges cost no measurable kernel time on p4.

The design's bank-affine split, bank-major range interface, block schedule and kernel are
unchanged; only the host knob defaults moved.

#### Kernel check for dtypes outside Phase 0

`test_g6_zeros_debug.py::test_dirty_destination_outside_phase0` runs p4 bfloat16 and p1 bfloat8_b
through the same program past `validate()`, over a poisoned range: every element reads back +0.0.
Adding those dtypes is a SUPPORTED-only change.

### Rebuild from the full-grid bank-run design (implementer)

Rebuilt against `op_design.md` (`bank_run_write`): bank-major position space, balanced contiguous
ranges from `ttnn.split_work_to_cores(..., row_wise=True)`, per-core RT `(q_begin, q_count)`,
common RT `(output address, P)`. One writer kernel (BRISC, NoC1) and one `cb_zero_src` on a single
`CoreRange` over the whole grid; cores without positions return before any NoC call. Each writer
zero-fills its source once (`Noc::async_write_zeros` loopback), maps its first run in the fill's
shadow, then writes each run with `noc_async_write_one_packet_set_state` / `_with_state` from that
constant source (a tail packet re-sets the length), and drains with one `noc_async_write_barrier`.
The kernel takes the bank count from `NUM_DRAM_BANKS`; the host never sends one. No reader or
compute kernel (the design's kernel set). SUPPORTED: `dtype` float32, `layout` TILE.

The design's plan with its Phase 0 knob values is still built and tested. Two knobs were added
next to the design's `grid` (`WRITER_GRID`), both RT-arg / core-order only, as the design's lamps
describe:
- `WRITER_CORES_PER_BANK`: how many cores of the rectangle take positions (`None` = all).
- `CORE_ORDER`: `row_major` (design Phase 0) or `bank_affine` (each bank's ranges go to the cores
  nearest its NoC1 endpoint, `get_optimal_dram_bank_to_logical_worker_assignment(NOC_1, (0,0))`).

#### Deviation from the design's Phase 0 knob values (measured)

Device kernel duration, ns, Blackhole p150 (8 DRAM banks, 11x10 grid), one fresh run per variant.
Every variant launches the same 110-core rectangle.

| variant | p1 bf8b 285.8 MB | p2 fp32 12.6 MB | p3 fp32 6.3 MB | p4 bf16 0.26 MB |
|---|---|---|---|---|
| design Phase 0: row_major, all 110 cores, Z=16K | 1,330,776 (215 GB/s) | 58,357 | 30,456 | 1,806 |
| row_major, 8 cores (rectangle 8x1), Z=16K | 902,833 | 40,824 | 20,959 | 1,965 |
| bank_affine, all 110 cores, Z=16K | 896,684 | 40,739 | 21,141 | 1,567 |
| bank_affine, 2 per bank (16), Z=16K | 1,106,341 | 52,964 | 26,793 | 2,342 |
| bank_affine, 1 per bank (8), Z=16K | 559,660 | 25,807 | 13,656 | 1,824 |
| bank_affine, 1 per bank (8), Z=8K | 559,700 | 25,661 | 13,412 | 1,622 |
| **bank_affine, 1 per bank (8), Z=4K (default)** | **559,553 (510.7 GB/s)** | **25,599** | **13,316** | **1,527** |
| bank_affine, all 110 cores, Z=4K | 858,562 | 39,037 | 20,595 | 1,505 |

- Bound: the DRAM banks. p1 writes at 510.7 GB/s against Blackhole's 512 GB/s nominal peak, so
  p1/p2/p3 have no data-movement headroom left. The p1 target (<= 837 us) is met only with the
  bank-affine single writer per bank; the design's Phase 0 values give 1331 us.
- Every extra writer per bank costs time: their NoC1 paths converge on one bank endpoint.
- `ZERO_SRC_BYTES_MAX` does not move the DRAM-bound shapes; it is fixed cost before the first
  write on p4, so the default is 4096.
- Defaults: `CORE_ORDER = "bank_affine"`, `WRITER_CORES_PER_BANK = 1`, `ZERO_SRC_BYTES_MAX = 4096`.
  The kernel rectangle stays one `CoreRange` over the grid (BB-17); with `P % NB == 0` each writer
  holds exactly one run.
- Mesh: one program runs on every chip, and the placement follows chip (0, 0)'s bank anchors. On a
  chip with different harvesting the writers are still correct, only possibly farther from their
  banks.

Lamps not built: second NoC (`writer_riscs = 2`) — p1/p2/p3 are at the DRAM peak, nothing to gain;
overlap of the fill with the first writes — the 4 KiB source already makes the fill 8 loopback reads.

#### Kernel reach outside Phase 0

`test_g6_zeros_bank_run_debug.py` runs the program past `validate()` over a poisoned range: p4
bfloat16, p1 bfloat8_b (1088 B pages straddle packets), rectangles below the bank count (up to 8
runs per core), 13 and 1001 pages (unequal banks, ranges straddling banks), both core orders and
1 / 2 / all writers per bank. Every element reads back +0.0. Adding bfloat16 and bfloat8_b is a
SUPPORTED-only change.

`test_g6_zeros_debug.py` targets the previous implementation's `Knobs` / `get_program` and no longer
imports; it is kept as a record. `test_g6_zeros_perf_knobs.py` runs against the new knobs through
the public entry point; its `BANKS_PER_CORE` case (every core walks every bank, the design's rejected
`page_id_split_bank_coalesced`) was dropped.

### Rebuild from the bank-major zero-run design (implementer)

Built the design's single regime `bank_run` with its Phase 0 values unchanged: one writer kernel
(BRISC, NoC1), no reader or compute kernel. `writers_per_bank = max(1, min(grid // N, ceil(P / N)))`
gives 13 writers per bank, 104 cores in 2 row-wise rectangles. Writer `w` serves bank `w mod N` and
writes a contiguous slot run from a 16 KiB `cb_zero_source`, filled once per core by NoC loopback.
Each write is at most 16 KiB, there is one barrier per core, and no CB handshake. SUPPORTED: `dtype`
float32, `layout` TILE.

Live knobs, each defined once in `g6_zeros_program_descriptor.py`:
- `WRITERS_PER_BANK` (None means the Phase 0 rule)
- `ZERO_SOURCE_BYTES` (16384)
- `ZERO_SOURCE_REGIONS` (1)
- the per-core block count, which reaches the kernel as the runtime arg `num_blocks`

Plan check (`describe_plan`, probe_014): the per-core slot counts match the design's table for
p1–p4. Bank runs cover `[0, P)` exactly once for the production page counts and for ragged counts
1, 5, 8, 9, 103, 105 and 1000003.

Device kernel duration (Blackhole, one fresh `--profile` run each):

| case | cores | ns | DRAM write rate |
|------|-------|----|-----------------|
| p2 (16,12,128,128) fp32 | 104 | 49,884 | 252 GB/s |
| p3 (8,12,128,128) fp32 | 104 | 24,917 | 252 GB/s |

That is about 49 % of the 512 GB/s nominal DRAM peak, so p2 and p3 are not at the DRAM bound with
these values. The design's grid-sync lamp (`writers_per_bank` in {1, 2, 4, 8, 13}) and its
core↔bank placement lamp are the next things to measure. Both are knob turns with no kernel change.

## Refinement 1 — Numerical formats: bfloat16 and bfloat8_b
- Date: 2026-11-28
- What was done: widened `SUPPORTED["dtype"]` to `[float32, bfloat16, bfloat8_b]`. No kernel, program-descriptor or `validate()` change: the page stride comes from `ttnn.tile_size(dtype)` (cross-checked against the allocator's `buffer_aligned_page_size()`), the `cb_zero_source` format tag follows `output.dtype`, and the writer kernel is dtype-agnostic (it streams all-zero bytes; a bfp8 page of all-zero bytes is +0.0 in every datum and exponent). 1088 B bfp8 pages tile the 16 KiB chunks because chunks are byte-addressed over a contiguous bank run, not page-aligned. Reused: the single `bank_run` regime, plan, CB and writer kernel. Added: nothing. No `compute_kernel_config` is exposed: the op has no compute kernel. No EXCLUSIONS entry: every production cell is tile-aligned, so the usual bfloat8_b × non-aligned exclusion does not arise.
- Accuracy achieved: exact. p4 (1,4,256,128) bfloat16 and p1 (32831,1,32,256) bfloat8_b: max ULP 0, 0 negative zeros, max abs error 0.0 (PCC 1.0 and relative RMS are degenerate against an all-zero reference; the bit-pattern gate is the meaningful one).
- Golden test progress: the 6 previously `xfail_expected` cells (p1 bf8b ×3, p4 bf16 ×3: plain, model profile, dirty destination) now pass, 6/6 in the `-k bfloat` slice; with the 13 float32 cells unchanged this is 19/19 expected. Unit suite: 26/26 pass, 0 xfail (`test_g6_zeros.py`, `test_g6_zeros_precision_baseline.py`), including both dirty-destination cases.
- Issues encountered: None.
- Tests added: none. The existing acceptance and precision-baseline tests already parametrize p4 bf16 and p1 bf8b and gate on SUPPORTED, so they switched from xfail to run. The skill's `test_<op>_precision_matrix` (math_fidelity × fp32_dest_acc_en) does not apply: there is no compute kernel and no compute config; `test_g6_zeros_precision_baseline.py` is the op's precision characterization.
