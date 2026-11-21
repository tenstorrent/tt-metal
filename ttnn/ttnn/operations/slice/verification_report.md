# Verification Report: slice

Target: Quasar (`quasar`, 8×4 clusters) on the ttsim simulator (untimed). Specification: `eval/specs/slice/spec.py` (sha256 953c36e9…). Design: `op_design.md` (regime `page_relay`).

## Code Review

Overall: the Phase 0 implementation matches the design on every binding dimension. The fixes below are small; no architectural rework was needed.

**Fixed in place**
- `slice_program_artifacts.py`: the ring's kernel-side accessor name was a second `"slice_pages"` literal next to `RING` (`producer_of(RING, "slice_pages")`, `consumer_of(...)`). Both now use `RING`, the single source for the DFB id and the `dfb::slice_pages` accessor.
- `test_slice_plan.py`: added `test_a_tt_metal_program_follows_explicit_sync` at plans (1,1) and (2,2). The existing session-order test ran a tt-metal program only after the **implicit**-sync path. The explicit-sync path (row-major pages > 8 KiB) serves YOLOv8's traced `[1,144,8400]` call, and nothing ran after it in the same session. The new test runs slice → `to_layout(TILE)` → `to_layout(ROW_MAJOR)` and checks the result is exact. It passes.
- `op_design.md`: added the implementer's 8 KiB implicit-sync entry cap to the *Mechanism caps* table. The code had it but the design did not, so the next stage would have planned against a design that disagreed with the code. Also added a verifier note on how the DFB runtime picks txn-id counts (below). `l1_ledger.md`: appended the verifier audit.
- Added `test_slice_precision_baseline.py` (below).

**Checked and found conformant (no change)**
- Algorithm and pipeline: one DM→DM ring per cluster, page map on the reader (odometer, no per-page divmod), contiguous output pages on the writer, no compute kernel. The bytes never pass through a NEO or DEST (R1).
- Work distribution: `split_work_to_cores(grid, num_out_pages, row_wise=True)` over the queried grid. Contiguous `out_page` ranges fill all 32 clusters down to one page per core (26/32 on the smallest YOLO call).
- Both dataflow halves batched: the implicit path has no per-page handshake (the ISR credits one half-ring batch). The explicit path does one reserve/barrier/push per block on the reader and one wait/barrier/pop per block on the writer.
- Blocking-model fidelity: `DM_THREAD_PLAN`, `RING_DEPTH_BLOCKS` and `block_pages` are each defined once. The ring's `num_entries`, `thread_block_pages` and the divisibility rounding are all derived from them. No kernel loop bound is a literal, and every shape-dependent value is a runtime arg. The per-page NoC call is the mechanism itself (one interleaved page = one transaction), not a half-turned knob.
- Thread plan (read off the host program): reader `num_threads=R`, writer `num_threads=W` from the single constant `DM_THREAD_PLAN = (1, 1)`, passed as CT args (`reader_threads`, `writer_threads`). Kernels deal pages by `get_my_thread_id()`. The design gives (1,1) with the reason "bound by data movement: zero compute per page" (target page, *Choosing the plan*). Accepted. The test showing every thread ran (`test_every_thread_runs`: two calls, fresh outputs, plans (1,1), (2,2), (2,1), (3,3), (4,2)) and the session-order tests are present and pass. Measuring the plan against its alternatives does not apply: ttsim has no timing model, and `pipeline_logs/model_cycles.jsonl` does not exist for this run.
- Ring protocols: the implicit path issues tagged `NocOptions::TXN_ID` calls with no reserve/push/barrier. The explicit path lists the ring in `disable_dfb_implicit_sync_for` and tags every transfer with user trid 0. Both kernels hand trid 0 back as their last statement.
- API usage: `void kernel_main()`, `api/dataflow/...` includes, `TensorAccessor`. No `InterleavedAddrGen`, no Metal 1 `ProgramDescriptor` (refused on Quasar).
- Helper usage: no compute phase. The design's rejected-helper table holds: `copy` would push values through DEST, which breaks R1 for fp32/uint32, and no DFB page-relay dataflow helper exists. No mcast is needed (no shared operand).
- Specification rules: R1 (no element passes through another dtype; bit-exact, measured), R2 (negatives normalized before the support check), R3 (lists/tuples; loose case passes), R4 (TILE edge tiles copied whole; padding unspecified), R5 (DRAM and L1), R6 (`UnsupportedAxisValue` / `ExcludedCell`, both `NotImplementedError`), R7 (output `TensorSpec(..., buffer_type=input's)`, interleaved; `test_output_in_the_inputs_buffer` reads `output.memory_config()`), R8 (`eval.memory.memory_axes` gates `memory_config`/`sharding`). REJECTIONS X1–X3 raise `ValueError` (golden `test_rejection[X1..X3]` pass). One-dispatch rule: every call, including the whole-tensor copy, is exactly one `ttnn.generic_op`.

**Deferred** (to refinements; reasons in `op_requirements.md`)
- Ring sized to the busiest core's whole share (`block_pages = ceil(max_core_pages/2)`): folded into Refinement 3.
- Explicit-sync path drains the NoC once per block: Refinement 5.

## Registry Conformance
- `INPUT_TAGGERS = AXES` imported from `eval.specs.slice.spec`; no op-local axis function.
- `SUPPORTED` equals the spec's `PHASE0` and covers every gated axis: dtype, layout, memory_config, sharding, and the four `AXES`.
- `EXCLUSIONS = []`. `validate()` checks the malformed-call rejections first, then normalizes, then SUPPORTED per axis, then EXCLUSIONS, then the mechanism caps (burst cap / ring fit, `NotImplementedError`). The public `slice()` calls `validate()` on its first line.
- No `INVALID` in the op file.
- No auto-fixes to SUPPORTED: no XPASS evidence (`xpass_drift = 0`).
- `python3 -m eval.validate_contract slice --phase0`: PASSED, 0 errors. The 5 unresolved items are the spec's own UNRESOLVED markers.
- INVALID audit (spec): `INVALID = []`. Correct: TARGET has no block-float dtype, so the canonical bf8b + ROW_MAJOR entry does not apply. There are no cross-tensor couplings (single input) and no norm-like weight axes. bfloat8_b/bfloat4_b are not in TARGET (target page: no path on Quasar). Nothing to report.

## L1 Ledger Audit
- **Currency**: one DFB (`slice_pages`, entry `input.buffer_aligned_page_size()`, `num_entries = RING_DEPTH_BLOCKS × block_pages`) and one ledger row. The expressions match the code. The implementer's explicit-sync path adds no buffer and is recorded in the ledger's implementation note.
- **Capacity vs live set**: capacity equals the declared live set. Over-direction finding: the block is half the busiest core's share, so the ring spans the whole share (≤ 1 MiB cap) instead of a fixed streaming window. On the implicit path the ISR credits half the ring per batch, so the writer starts only after half the core's pages have landed. The size is bounded, with no OOM risk (≤ 67 KB, ≤ 1.7 % of the 3.96 MB budget on INPUTS). Not fixed in place; folded into **Refinement 3** (cap `block_pages` independently of the share). Under-direction: none.
- **Page format vs DEST**: n/a. The payload is dataflow-only and never enters DEST.
- **Disjoint lifetime**: n/a (one buffer).
- **Bounds / closed form**: `entry_bytes ≤ 65,536` (validate), `block_pages ≤ block_pages_cap`, total `RING_DEPTH_BLOCKS × block_pages × entry_bytes ≤ ring_cap_bytes`, all closed form.
- **Data-movement budget**: present and consistent with the split. Each block page is read once and written once, there is no cross-core traffic, and the cheapest split (all tie on bytes) is implemented.
- **Block-size defaults**: interleaved default held: the full grid is spread first, then the coarsest block for depth 2. Refinement 3 revisits that block size on overlap grounds (catalog `double_buffer`), as a co-tune, not a fit problem.
- **Per-core footprint**: `RING_DEPTH_BLOCKS × block_pages × entry_bytes`. It scales with `block_pages` (block knob), `RING_DEPTH_BLOCKS` (depth knob) and `entry_bytes` (dtype × W for RM, dtype for TILE).
- **Verifier finding on the depth knob** (now in `op_design.md`): the DFB runtime gives an implicit-sync ring the smallest txn-id count `n ∈ [2,4]` dividing `num_entries / max(R,W)`. With an even ring `n = 2` regardless of `RING_DEPTH_BLOCKS`, so the credited batch is always half the ring. On the implicit path `RING_DEPTH_BLOCKS` therefore does not create finer in-flight blocks by itself.

## Precision Baseline

`tests/ttnn/unit_tests/operations/slice/test_slice_precision_baseline.py`, bfloat16, seed 0, standard normal inputs. PCC via `comp_pcc`, abs deltas cross-checked with `comp_allclose`, relative RMS = RMS error / std(reference). Ratio = got/true over finite non-zero reference elements.

| Case | Shape, block | Layout / placement | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | got/true ratio p5 / median / p95 |
|------|--------------|--------------------|-----|-------------|--------------|------------------|-----------------------------------|
| small_tile_dram | (1,64,128), rows 0–32 | TILE / DRAM | 1.0 | 0.0 | 0.0 | 0.0 | 1 / 1 / 1 |
| medium_tile_dram | (4,128,512), rows 32–96 | TILE / DRAM | 1.0 | 0.0 | 0.0 | 0.0 | 1 / 1 / 1 |
| yolo_tile_l1 | (1,144,400), rows 64–144 | TILE / L1 | 1.0 | 0.0 | 0.0 | 0.0 | 1 / 1 / 1 |
| medium_rm_l1_wna | (4,128,47), rows 64–100 | RM / L1 | 1.0 | 0.0 | 0.0 | 0.0 | 1 / 1 / 1 |
| large_yolo_rm_dram | (1,144,8400), rows 64–144 (explicit-sync path) | RM / DRAM | 1.0 | 0.0 | 0.0 | 0.0 | 1 / 1 / 1 |

**Assessment**: bit-exact on every case (`torch.equal`). The ratio spread is a single point at 1.0, so there is no scale or structural error. This is what the design predicts: bytes are moved by the NoC only.
**Recommended tolerances**: exact. Every cell resolves to `exact` in `NUMERIC_RESOLVED` (bfloat16, float32 by entry; uint32 by class default), and Refinement 4 must keep float32/uint32 exact.

## Verifier CLI Summary
Golden run: `eval/eval_test_runner.sh eval/golden_tests/slice/ /tmp/slice_golden_final` (8 sim workers, 484 tests, 0 hangs). Report copied to `ttnn/ttnn/operations/slice/verifier_report.json`.
- supported_pass: 69 (68 PHASE0 cells + loose `bounds_as_lists`)
- xfail_expected: 412
- invalid_skipped: 0
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0
- no_axes_found: 3 (`test_rejection[X1..X3]`, all pass)
- Queue check (`--queue op_requirements.md`): 6 values in TARGET − SUPPORTED, 0 gaps.

Unit tests: `scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/slice/` gives 84 passed: acceptance 13, thread plan / session order / R7 66, precision baseline 5.

**Probe evidence for the queue** (`tests/ttnn/unit_tests/operations/slice/probes/probe_005.py`; SUPPORTED widened in-process only). On the Phase 0 kernels, 11 of 13 cases are exact: rank 2 (TILE, RM), rank 4 (TILE `h`, RM `outer`), TILE `w` with tile-aligned start (incl. W 4096 in L1), RM H start inside a tile (rank 3 and 4), float32 (TILE; RM rows of 33,664 B on the explicit path; RM `W=47` in L1). uint32 (TILE and RM) fails on the host with `TT_FATAL dfb.cpp:148 is_data_format_supported`: Quasar refuses uint32 as a DFB format. These results set the bundling: the validate-only halves ride with the scheme-changes that complete their axis (R1, R2), and dtype + rank form one cheap bundle (R4).

## Recommendations
- **Queue shape** (see `op_requirements.md`):
  - R1 `rm_column_cut` + TILE `w` (adds `sliced_region=w`)
  - R2 `tile_retile` + RM H start (adds `start_alignment=non_tile_aligned`; depends on R1 because every TILE W-shift cell is `sliced_region=w`)
  - R3 perf: ISR batch granularity
  - R4 float32 / uint32 / ranks 2, 4
  - R5 perf: trid rotation on the explicit-sync path

  No `LOOSE_CASES` entry is perf-flagged, so both perf targets are free-selected, and the anchor rule puts no constraint on R1/R2. The YOLO traced calls are already supported.
- **R2 mechanism conflict**: a compute-stage retile (untilize → shift → tilize) would pass values through DEST, which conflicts with R1 once float32/uint32 land. The DM-hart face shuffle is preferred unless the compute path is proven bit-exact for all three dtypes.
- **Perf lamps not filed** (no structural count would settle them on an untimed backend): grid occupancy on tiny outputs (26–39 pages on 26–32 clusters, one page each; a `MIN_PAGES_PER_CORE` knob), `row_wise=True` vs `False` placement on Quasar's 8×4 grid with 2 DRAM channels, and a (2,2) thread plan for issue-bound small RM pages (128-B entries). Revisit on `ttsim-perf` or hardware.
- **Spec open items** (not defects; nothing in the suite depends on them): `NUMERIC["special_values"]`, `NUMERIC["freedoms"]`, `PERFORMANCE`, `DEFINITION["tt_regen_spec"]["governs"]` are UNRESOLVED. An empty block (`end == start`) is unspecified; the op refuses it with `NotImplementedError`, and no cell exercises it.
