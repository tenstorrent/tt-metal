# Verification Report: g6_slice

Phase 0 (bfloat16, TILE, DRAM interleaved, host-list begins, unit steps; regimes `page_copy`, `row_runs`,
`col_realign`). Board: Blackhole, 110-core compute grid (11 × 10), 8 DRAM banks.

## Code Review

Fixed in place:

| # | Finding | Fix |
|---|---------|-----|
| 1 | **DRY.** The descriptor re-derived every tile quantity (`tensor_in/out_ht/wt_tiles`, `rows_last_tile_row`, `cols_last_tile_col`, begin tile/offset splits) from a second literal `tile = 32`, next to the same derivations in `_plan()`. Two sources for one value. | All derived quantities are computed once in `_plan()` and the descriptor only reads them from the plan. The face-half test uses a named `FACE_DIM`. |
| 2 | **Over-capacity CB (ledger audit 2).** `cb_slice_pages` was always `CB_DEPTH × block_pages`. A core with a single block (every case under ~3.5 k output pages, e.g. 587: 6 pages per core) never has block k+1 in flight, so the second slot was never live. | `cb_depth = min(CB_DEPTH, num_blocks_max)`. The L1 bound on `block_pages` is still solved with `CB_DEPTH`. 587 drops from 24 KB to 12 KB per core. `describe_plan` reports the effective depth. |
| 3 | **BB-21: work outside every zone.** The reader's and writer's run-list build ran before the first zone. In the zone build that is ~710 cycles per kernel that reads as waiting. On the reader it sits on the critical path (≈10 % of 587's kernel). | `G6_SETUP` zone added to both kernels around the run-list build. The plain build is unchanged. Zone list in `op_design.md` updated. |
| 4 | A host (non-device) `input_tensor` failed with `AttributeError` deep inside planning. | Structural `ValueError("... must be on device")` is now raised first in `validate()`. |
| 5 | Coverage gaps: `col_realign` was never run with more than one block per core, so its reader-private right-column scratch was never reused across blocks. Nothing pinned the depth clamp or the host-tensor refusal. | `test_several_blocks_per_core` now includes `rule_w_and_h_begin_in_tile` (`MAX_CORES = 2`, `BLOCK_PAGES_CAP = 3`: 2 blocks per core). Added `test_single_block_cores_allocate_one_slot` and `test_host_tensor_is_refused`. |

Checked and clean (no change needed):

- **Kernels.** Syntax is `void kernel_main()`. Includes use `api/dataflow/dataflow_api.h`. All addressing goes through `TensorAccessor`, with no `InterleavedAddrGen`. CB sync is balanced: one `cb_reserve_back` and one `cb_push_back` of `block_pages` per block in the reader, one `cb_wait_front` and one `cb_pop_front` of `block_pages` per block in the writer, and a ragged last block narrows only its NoC work. The writer calls `noc_async_writes_flushed()` before every pop (BB-06) and `noc_async_write_barrier()` at the end (BB-03). The `copy_tile_column<cb, T>` element type is derived from `page_bytes`.
- **Helpers.** No compute kernel exists, so there is no compute helper to use. `fetch_block` and `store_block` stay raw NoC: `op_design.md` lists every dataflow helper considered and why each cannot gather byte runs from a `TensorAccessor` page sequence into a CB block. No multicast or semaphores exist. Broadcast checks don't apply.
- **Prompt rules** (`eval/prompts/g6_slice.txt` `## Rules`):

  | Rule | Status |
  |------|--------|
  | One program, no ttnn op around the copy | One `generic_op` per call |
  | No host-side transforms | None |
  | Bits copied in the input's own format | Bits never leave the NoC except the in-L1 RISC-V column moves of `col_realign` |
  | Only pages holding selected elements are read | `row_runs` reads only selected face-row runs. `col_realign` reads only the face halves holding selected columns of ≤ 2 tiles. No whole tile rows are read to drop columns |
  | `h_non_aligned` in SUPPORTED and never in EXCLUSIONS | Holds |
  | Core count set by the blocking decision | `split_work_to_cores` over the full grid; `MAX_CORES` knob |
  | Golden shapes never refused for size | No OOM on any golden shape |
  | Unused paths compiled out per signature (RB-038) | CT regime flags plus `if constexpr` |
  | Model-wide rules | BB-01/02/03/05/06/07 (plan cache keyed on the full signature; begins are RT, so one compiled program serves every begin) all hold. BB-08 (`describe_plan`), BB-16 (trace replay is bit-exact once the program cache is on, see Harness below), BB-17 (row-wise split gives ≤ 2 rectangles) and BB-21 (after fix 3) hold |

  No MUST violation. Advisory "prefer to compile out paths" is followed.

Deferred to refinements (architectural, not fixable here):

- Block formats (`bfloat8_b`, `bfloat4_b`) do not compile even for `page_copy`. `TileGeometry`'s and the col-realign `static_assert`s fire inside discarded `if constexpr` branches because `kernel_main` is not a template. This is Refinement 4.
- `out_l1_bytes_per_core = 0` is hard-coded in `_plan()`. That is correct for DRAM `out` only, and Refinement 2 must derive it from the placement (BB-05).

## Design Conformance

| Dimension | Design | Implementation | Verdict |
|-----------|--------|----------------|---------|
| Algorithm | Three regimes chosen on the host from `(b4, M4)`. Rows land at natural in-tile offsets, and the writer applies the rotation | Same selection function. Regime is pinned per acceptance case via `describe_plan` | ✓ |
| Pipeline | Reader on NCRISC/NoC0 → `cb_slice_pages` → writer on BRISC/NoC1. No compute kernel | Same | ✓ |
| Work distribution | Contiguous output-page ranges over the **full** grid, `row_wise=True` | 110/110 cores on every case ≥ 110 pages, e.g. 703: 2532–2533 pages per core | ✓ fills the machine |
| Batching | 1 handshake + 1 barrier per block on **both** halves | Reader: reserve, all reads, one barrier, push. Writer: wait, all writes, one flush, pop | ✓ both halves block-expressed |
| Block knobs | `block_pages` (CT), `cb_depth`, `MAX_CORES` | One source each: `_plan()` → CT arg / CB size. Overridable at call time | ✓ (DRY fixed, item 1) |
| Inter-core communication | none | none | ✓ |

## Registry Conformance

- `INPUT_TAGGERS`: 7 taggers, all `(inputs, axes)`, reading only the scenario dict.
- `SUPPORTED`: every axis the kernels gate on, plus every tagger key.
- `EXCLUSIONS = []`.
- `validate()` order: structural `ValueError`, then taggers, then `UnsupportedAxisValue`, then `ExcludedCell`. It is the first line of `g6_slice()` and `describe_plan()`.
- The op file declares no `INVALID`.
- Auto-fixes to SUPPORTED: none needed (0 `xpass_drift`).
- INVALID audit (`feature_spec.py`): 2 entries, `{bfloat8_b, ROW_MAJOR}` and `{bfloat4_b, ROW_MAJOR}`.
  - Both describe one tensor (x).
  - Both are structural, since a block format has no row-major form.
  - The canonical `bfloat8_b + ROW_MAJOR` entry is present.
  - There are no cross-tensor couplings and no "not yet" entries.
  - `invalid_skipped = 0` is expected: `INPUTS` is the fixed production list, which iterates no axis, and no production case is a block format in ROW_MAJOR.

## L1 Ledger Audit

- **Currency.** Both CB rows match the descriptor. `cb_slice_pages = cb_depth × block_pages × page_bytes`, with `cb_depth` now clamped (row and symbol table updated). `cb_right_col_tile = block_pages × page_bytes`, only in `col_realign`.
- **Capacity vs live set.**
  - Over: one hit, the single-block second slot, fixed in place (Code Review 2). No other CB exceeds its live set.
  - Under: `cb_slice_pages` spans `out_page` with capacity `block_pages × cb_depth`, and its capacity scales with the block axis it spans. `cb_right_col_tile` holds one right-column page per page of the block between fetch and realign. That is the live set, consumed by the same thread before the next fetch, so it is correctly one block deep.
- **Page format vs DEST.** No compute kernel exists. Both CBs are dataflow-only payloads in `x.dtype`, so neither direction applies.
- **Disjoint lifetimes.** The only pair (`cb_slice_pages`, `cb_right_col_tile`) is concurrently live during `realign_columns_block`. The justification is present.
- **Bounds.** Every symbol has a bound and a predicate. The total is closed-form: `L1_op = (cb_depth + W_SHIFT) × block_pages × page_bytes + 64 × num_cbs ≤ L1_budget − out_l1_bytes_per_core`. Nothing scales with `N` or `M` except through `block_pages ≤ core_pages_max`, which is clamped by `block_pages_cap`.
- **Data-movement budget.** Present and consistent with the code.
  - Each selected byte crosses DRAM once each way in `page_copy` and `row_runs`.
  - `col_realign` reads ≤ 2× its selected face halves. The cheaper `col_realign_sliding` is a `deferred` regime row with a positive reason: no production TILE begin on the last dim lies inside a tile.
  - The cheapest-traffic split (contiguous output-page ranges) is the one implemented, and every alternative ties on bytes.
- **Block-size defaults.** Interleaved: full grid first, then the coarsest block that fits (329 pages bf16). The default `BLOCK_PAGES_CAP = 32` departs from it. Re-measured with a same-session paired A/B at 1350 MHz, 3 reps per arm:

  | Case | cap 32 vs cap 329 |
  |------|-------------------|
  | 703 | 32 is 1.5 % faster (2669.6 vs 2710.4 µs) |
  | 795 | 32 is 1.4 % faster |
  | 1112 | 32 is 2.1 % faster |
  | 179 | 32 is 1.2 % faster |
  | 730 | tie |

  All spreads are < 0.5 %. The edge is reproducible but under the campaign's 2–3 % win threshold, so it is recorded as a tie that costs 10× less L1 (131 KB vs 1.35 MB per core), not as a measured speedup. It is kept, with the justification corrected in the ledger.
- **Per-core footprint (Phase 0 default).**

  | Case | Bytes per core |
  |------|----------------|
  | `page_copy` / `row_runs` | 131,072 B |
  | `col_realign` | 196,608 B |
  | Single-block cases | `block_pages × 2048` B |

  `cb_slice_pages` scales with `block_pages × cb_depth`. `cb_right_col_tile` scales with `block_pages`.

## Precision Baseline

`tests/ttnn/unit_tests/operations/g6_slice/test_g6_slice_precision_baseline.py`, bf16, randn input. PCC is `comp_pcc`, max-abs is `comp_allclose`, ULP is on the bf16 bit pattern, and the ratio is got/true over non-zero references.

| Shape (regime) | Out | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | Max ULP | got/true median [p5, p95] |
|----------------|-----|-----|-------------|--------------|------------------|---------|---------------------------|
| (1,1,128,64)[..,17:82,:] (`row_runs`, two input tiles per output tile) | (1,1,65,64) | 1.0 | 0 | 0 | 0 | 0 | 1 [1, 1] |
| (1,1,96,128)[..,17:90,20:120] (`col_realign`) | (1,1,73,100) | 1.0 | 0 | 0 | 0 | 0 | 1 [1, 1] |
| c179 (1,2304,3072)[:, :2048] (`page_copy`) | (1,2048,3072) | 1.0 | 0 | 0 | 0 | 0 | 1 [1, 1] |
| c587 (16,4096,1280)[:, 4095:] (`row_runs`, offset 31) | (16,1,1280) | 1.0 | 0 | 0 | 0 | 0 | 1 [1, 1] |
| c1112 (16,1024,4160)[..., :2560] (`page_copy`) | (16,1024,2560) | 1.0 | 0 | 0 | 0 | 0 | 1 [1, 1] |

**Assessment.** Bit copy in every regime. NaN and inf pass through: the acceptance case and the golden `nan_inf` loose case are both bit-exact. There is no scale signature, since the ratio is exactly 1.

**Recommended tolerances.** Exact, matching `helpers.TOLERANCES[bfloat16] = "exact"`. Any non-zero ULP is a bug: a format conversion or a misplaced value.

## Verifier CLI Summary

Run: `EVAL_CAPTURE_L1=0 eval/eval_test_runner.sh eval/golden_tests/g6_slice/ <dir>`, then `python3 -m eval.verify_supported`. The artifact is `ttnn/ttnn/operations/g6_slice/verifier_report.json`, with `supported_pass` entries trimmed to node IDs to fit the repo's 500 KB file limit. 1349 collected, 900 passed.

- supported_pass: 881 (868 production + 13 loose)
- xfail_expected: 449 (442 production + 7 loose). Every cell maps to a refinement except `step2` (`step_mode="strided"`, deliberately not in TARGET).
- invalid_skipped: 0
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0
- no_axes_found: 19. These are the `test_regression.py` numerics tests outside the registry, and all pass.

xfail cells by refinement:

| Axis values | Cells | Refinement |
|-------------|-------|------------|
| bf16 ROW_MAJOR DRAM | 190 | R1 |
| bf16 TILE `l1_interleaved` | 22 | R2 |
| bf16 TILE `l1_width_sharded` | 3 | R2 |
| bf16 TILE `device_tensor` | 7 | R2 |
| bf16 ROW_MAJOR `l1_interleaved` | 1 | R1 + R2 |
| float32 TILE | 98 | R4 |
| int32 ROW_MAJOR | 110 | R1 + R4 |
| uint32 ROW_MAJOR | 11 | R1 + R4 |
| bfloat4_b TILE DRAM | 4 | R4 |
| bfloat8_b TILE `l1_interleaved` | 2 | R2 + R4 |
| strided | 1 | none (not in TARGET) |

### Harness findings (not op defects; for the eval owners)

1. **The default golden run cannot pass the trace loose case, and it takes the rest of the suite down with it.**
   - `eval_test_runner.sh` defaults to `EVAL_CAPTURE_L1=1`. `eval/l1_profiling.py:start_for_item` then calls `device.disable_and_clear_program_cache()` for every graded test.
   - `test_op_loose[Qwen703_trace]` captures a trace. With the cache off, capture hits `TT_FATAL: Cannot load new binaries during trace capture` for any implementation.
   - `helpers.run_scenario` does not wrap `begin_trace_capture`/`end_trace_capture` in `try/finally`, so the device stays in capture mode. Every later test fails with `Writes are not supported during trace capture`, and re-opening the module device for `test_regression.py` segfaults (pytest exit 139, no JUnit XML).
   - The runner then reports `PASSED=0 … HANGS=1`. With `EVAL_CAPTURE_L1=0` the same tree gives 900 passed / 0 failed.
   - Suggested fix: keep the program cache on, or skip L1 capture for `trace_replay` cases, and close the capture in a `finally`.
2. **Board clock moves between sessions.** The profiler's `CHIP_FREQ` read 800, 1112 and 1350 MHz in different sessions of this run. 703 measured 2535–3151 µs on the same build across sessions. Device-ns comparisons are only valid inside one session (interleaved A/B) or normalized to the logged clock. The implementer's recorded 2534 µs fits a ~1000 MHz session.

## Performance Baseline

Same-session, 1350 MHz, 3-rep medians, default knobs: 703 2669.6 µs, 795 1336.7 µs, 1112 395.7 µs, 179 62.5 µs, 730 21.8 µs, 587 6.9 µs, 1175 10.6 µs.

Same-session pair against stock `ttnn.slice`, timed as a black box (one session at 1112 MHz):

| Case | Stock | g6_slice | Ratio |
|------|-------|----------|-------|
| 703 (p1, flagged) | 3767 µs | 3147 µs | **1.20× faster** |
| 795 | 1879 µs | 1577 µs | 1.19× faster |
| 730 | 35.7 µs | 26.3 µs | 1.36× faster |
| 587 (single row, 640 out tiles) | 2.68 µs | 8.70 µs | **3.2× slower** |

- **Flagged case 703.** At 1350 MHz it runs 2669.6 µs. That is 1.078× the lead's 2475.8 µs roofline, under the 1.35× limit (3342 µs), and below the lead's stock figure (2930.5 µs). About 7 % of DRAM-bound headroom remains (Refinement 3).
- **Single-row `row_runs` class (587, 77 production cases with a dim r−2 begin inside a tile).** This loses to the incumbent. It is not a core-count problem: 8 / 16 / 32 / 64 / 110 cores give 12.1 / 10.1 / 8.0 / 7.5 / 6.9 µs.
  - 587 zone profile, 1350 MHz, p50 per core: reader `G6_SETUP` 706 cycles, `G6_FETCH_ISSUE` 746 (12 reads of 32 B), `G6_READ_BARRIER` 3965. Writer `G6_CB_WAIT` 4882.
  - The time is DRAM request latency and rate on 32-B face-row reads (2 per tile), plus the setup in front of the first read.
  - See Recommendations.

## Recommendations

- **Refinement order.**
  1. R1 ROW_MAJOR (`stick_copy`)
  2. R2 L1 placements + device-resident begins
  3. R3 perf on 703
  4. R4 element formats
  5. R5 perf on 703

  The perf-1 contract of 703 (bf16, TILE, DRAM, host list, no compute config) is already fully in SUPPORTED, so R1/R2 are ordered by difficulty, not by the anchor rule.
- **Single-row class vs stock (open, not queueable).** Every perf refinement must target the flagged 703 (`feature_spec` flags only 703), so this deficit has no queue slot. Two levers fit the 587 zone profile:
  1. Coalesce the two face-column runs of a row band into one read whenever the bytes between them are dead in the staging page. For a single row that is one 544-B read instead of two 32-B reads per tile, and DRAM cost here tracks requests, not bytes.
  2. Precompute the ≤ 4 run lists on the host (common RT args), taking the ~700-cycle `G6_SETUP` off the reader's critical path.

  Suggest the `feature_spec` owner flag loose case 587 (`row_in_tile`) for a perf pass. Meanwhile R3 and R5 carry 587 and 1175 in their no-regression guard sets.
- **L1.** No golden shape is near the budget. L1 placement (R2) is the first change that can approach it, because `out` bytes then count against BB-05.
- **Block formats.** The geometry `static_assert`s must move behind regime-specific instantiation before any 1088/576-B page can build (R4).
