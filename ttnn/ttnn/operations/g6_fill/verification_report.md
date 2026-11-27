# Verification Report: g6_fill

Board: Blackhole p150b (11 x 10 compute grid). Code verified: the working tree committed with this report.

## Verdict

- The op is correct on every supported cell: bit-exact, one program per call, input never read, 64 B of L1 per core.
- The graded golden run has **one `supported_fail`: the trace loose case (`loose6`)**. The harness causes it, not the op: the graded L1 capture disables the program cache for the whole test body, and no program can be captured into a trace without the cache. With L1 capture off, all 16 supported cells pass. This is escalated below (Harness findings, H1) and is **not** queued as a refinement, because the op cannot fix it.
- `xpass_drift = 0`, `xfail_wrong_mode = 0`.
- The queue (`op_requirements.md`) has **one generality refinement** (int32 + ROW_MAJOR, regime R2). It has **no perf refinements**: `eval/prompts/perf_rules.txt` forbids measuring or targeting any cell of 32 tiles or fewer, and every g6_fill cell, the PERF FOCUS `[512]` included, is at most 16 tiles (H3).

## Code Review

Every item below is a fix that is now in the code, verified on device with plain, `--dev` (watcher + NoC sanitizer) and `--profile` (zone) builds.

| # | Fix | Why |
|---|-----|-----|
| 1 | Knob single source: a frozen `Knobs` dataclass built by `current_knobs()` is the only reader of `MAX_FILL_CORES`, `BLOCK_W_TILES`, `FILL_ROW_SOURCE` and `FILL_ROW_DEPTH`. Both the plan (`make_plan(tensor, knobs)`) and the descriptor cache key take that one object. | Before, `FILL_ROW_DEPTH` was read straight from the module in the CB sizing and was **missing from the cache key**. Turning it would have replayed a stale program (BB-07), and the knob list was restated in two places. |
| 2 | The `dst_addr` index in the common runtime args is now host-owned: CT arg 7 `dst_addr_arg = plan.fill_words`, the same value that sizes the common-args list. A `static_assert` checks that the in-place fill run precedes it. | The kernel used to re-derive `fill_words` from `segment_words` and the source flag, so the same layout was defined on both sides of the host/kernel boundary (DRY). |
| 3 | `FILL_ROW_SOURCE` default `"common_args"` → `"cb"` (the design's Phase 0 source). `"common_args"` stays a live setting. | With `"common_args"` the op allocates no L1. The graded runner (`EVAL_CAPTURE_L1=1` by default) raises `L1ProfilingError: peak_total=0` on that, and **0/19** golden cells passed (run on device, then reverted). With `"cb"` the recorded peak is 64 B on every cell. Cost: about 100 ns per call (implementer's measurement, table in `l1_ledger.md`). See H2. |
| 4 | The writer's `dst_addr`, `start_tile` and `core_w_tiles` reads moved inside the `G6FILL-ISSUE` zone. | BB-21: no work of a core may fall outside a named zone. The zone build compiles and emits all three zones. |
| 5 | `l1_ledger.md` rewritten for currency: it described `"common_args"` as the default and the program as CB-less, which no longer matched the code. | Ledger currency audit (below). |
| 6 | Removed an empty, untracked `probes/probe_003.py`. | Leftover of an interrupted session. |

Items 1–3 were uncommitted changes left in the working tree by an interrupted earlier verifier pass of this run, after the boundary commit. I reviewed them, checked the zero-peak claim against `eval/l1_profiling.py` and on device, and kept them.

Reviewed and left unchanged (correct as written):

- `kernel_main()` syntax, `api/dataflow/dataflow_api.h` include, `TensorAccessor` (no `InterleavedAddrGen`).
- No CB handshake exists, so there is no push/wait mismatch to fix. `cb_fill_row` is a private source buffer of the one writer thread.
- Non-posted `noc_async_write_one_packet` followed by a single `noc_async_write_barrier()` as the last action (BB-03).
- Helper usage: no kernel_lib helper writes an arbitrary 32-bit constant run to `(page, offset)` destinations of an interleaved tensor. The planner's rejected-helper table holds. Multicast is not applicable, because the scalar reaches every core as a common runtime arg. There is no compute phase.
- Broadcast efficiency: the 64-B run is the minimum NoC source for a 64-B segment write. No redundant fill.
- Calling `get_noc_addr(tile, offset)` once per segment instead of hoisting the tile base: inline, same page id, and only 2 calls per core on production shapes. Left as part of the write-issue-path lamp (Recommendations), since nothing here can be measured (H3).

Deferred: nothing.

## Prompt Rules (`eval/prompts/g6_fill.txt` → `## Rules`)

All rules are MUST/MUST NOT. Only the ones whose condition applies to the current SUPPORTED set are listed.

| Rule | Status | Evidence |
|------|--------|----------|
| Write `e` with a device program of its own; never read the input; no host build/upload | holds | `input_tensor` is not a `generic_op` operand and has no accessor. `test_input_untouched`, plus the input-unchanged check in every separate-output cell. |
| Every element is `e` bit for bit, all 32 bits | holds | Dataflow-only path, `fill_bits` is a `uint32` common arg. Exact `1/3, -0.0, 1e-40, ±inf, NaN, FLT_MAX, 2^24+1, 2^60+2^36+1` (acceptance, golden `loose4`). |
| Repeat call with a new value holds the new value, also from the program cache | holds | `test_new_value_reuses_cached_program` (no new cache entry, values `1.0, 1.0, 0.1, 1.0`). Golden `loose6` passes with L1 capture off (H1). |
| Write into `output_tensor`'s own buffer, return it; input unchanged otherwise | holds | `check_returned` on every cell; in-place `loose7`/`loose8`. |
| Non-aligned values handled natively, never in EXCLUSIONS | holds | Both row-0 segments of the last tile are always written. `EXCLUSIONS = []`. `[185]` checks positions 16–24 in face 1. |
| Program is the op's own kernels and descriptor | holds | One writer kernel under `kernels/`, built by `g6_fill_program_descriptor.py`. |
| Invalid `fill_value` → ValueError, before the support gate | holds | `test_refused_fill_values_float32`, golden `loose9`. Contract checks run before SUPPORTED in `validate()`. |
| OOM / BB-05 | holds | 64 B per core, constant in `N`. |
| BB-01 one program | holds | `device_num_programs = 1` on every single-call graded cell. |
| BB-07 no rebuild, complete cache key | holds | Descriptor cached on `(device, shape, dtype, layout, memory_config, Knobs)`. Per call only the common args are patched. Fix 1 closed the `FILL_ROW_DEPTH` gap. |
| BB-16 no allocation under trace | holds | Nothing is allocated per call. Trace replay is bit-exact (H1 run). |
| BB-17 few rectangular core ranges | holds | `[512]`: `(0,0)-(10,0)` + `(0,1)-(4,1)`. Every other shape: one row range. |
| BB-21 zones and role dump | holds | Three zones, each once per core (fix 4). `g6_fill_plan()` reports regime, ranges, tiles and role per core. |
| BB-22 production-only tests | holds | Tests use `[512] [256] [207] [196] [185]` and the rule-required value sets only. |

No soft (`prefer` / `consider` / `avoid`) rules exist in this prompt, so there are no advisories.

## Registry Conformance

- `INPUT_TAGGERS` holds `tag_dtype`, `tag_layout`, `tag_alignment`, all `(inputs, axes)`, with return values as the prompt specifies (`"tile_aligned"` never occurs for rank 1).
- `SUPPORTED = {"dtype": [float32], "layout": [TILE], "alignment": ["w_non_aligned", "h_non_aligned"]}` covers every axis the kernel gates on and every tagger key.
- `EXCLUSIONS = []`, as the prompt requires.
- `validate()` is the first line of `g6_fill`. It runs the contract checks (ValueError), then SUPPORTED per axis (`UnsupportedAxisValue`), then EXCLUSIONS (`ExcludedCell`).
- The op file does **not** declare `INVALID`.
- No SUPPORTED auto-fixes were needed: `xpass_drift = 0`.

**INVALID audit (`feature_spec.py`):** `INVALID = []`. There is no cross-tensor coupling and no "not yet" entry. The canonical bf8b + ROW_MAJOR entry does not apply, because TARGET has no `bfloat8_b`. g6_fill is not a norm, so there are no weight-canonicalization cells. The taggers only ever emit production `(dtype, layout)` pairs, so off-production pairs never become cells and need no INVALID entry. Well-formed.

## Design Conformance

| Dimension | Result |
|-----------|--------|
| Algorithm | Regime R1 as designed: two 64-B row-0 segments per tile at `{0, face_bytes}`, no padding writes, no compute kernel. |
| Pipeline topology | One writer kernel (BRISC, `WriterConfigDescriptor`). Build → issue → one barrier. |
| Work distribution | `w_tiles` split by `split_work_to_cores(..., row_wise=True)` over `min(tensor_w_tiles, grid)` cores. 16 / 8 / 7 / 7 / 6 cores for `[512] / [256] / [207] / [196] / [185]` (graded `device_num_cores` agrees). The grid is filled to the shape's independent extent; no core is idle that could take work. |
| Inter-core communication | None, as designed (R6 multicast rejected). |
| Blocking-model fidelity | `MAX_FILL_CORES`, `BLOCK_W_TILES`, `FILL_ROW_DEPTH` and `FILL_ROW_SOURCE` are knobs with one source (fix 1). `block_w_tiles` reaches the kernel as one CT arg derived from the plan. No CB scales with a whole-op dimension. Segment geometry is derived once from `tile.tile_shape` / `face_shape`. The knob sweep (`test_g6_fill_knobs.py`, 30 cases) stays bit-exact. |
| Expression | The scatter issues every write of a core without a per-unit barrier or handshake. The block factor only shapes the loop and adds no synchronization. Not an expression failure. |
| Axis accounting | `cb_fill_row` spans `lane` only and streams `w_tiles`, `row0_segment` and `plane_h`. Correct: one constant run serves every write. |

Stale design text (not a code issue): `op_design.md`'s cache-key row lists only `MAX_FILL_CORES`. The implementation keys on all four knobs, a superset, which is correct.

## L1 Ledger Audit

| Check | Result |
|-------|--------|
| Currency | **Fixed.** The ledger said `"common_args"` was the default with no CB. It now matches the code: `cb_fill_row`, 1 page of 64 B, allocated by default. Graded `device_l1_peak_bytes = 64` on every cell agrees. |
| Capacity vs live set | Capacity = live set = 1 page (the constant run). Not over, not under. |
| Page format vs DEST | `Float32` page with no DEST anywhere: a dataflow-only payload that must keep all 32 bits (contract). Justified. |
| Disjoint lifetime | Single CB. Nothing to share. |
| Bounds / closed form | `fill_row_depth * segment_bytes` with `face_w = 16` and `element_size = 4` asserted on the host. Constant in `N`. |
| Traffic budget | Present and consistent with the code: `128 * tensor_w_tiles` B written, 0 read, 0 cross-core. The cheapest split is implemented, and traffic is invariant under every candidate split. |
| Block-size defaults | Interleaved default held: spread across the full grid, then the coarsest block (the whole per-core assignment, 1 tile on production shapes). The implementer's measured grid lamp confirms full spread is fastest (`[512]`: 560 ns at 16 cores vs 586 / 670 / 881 / 1,327 ns at 8 / 4 / 2 / 1 core). |

Per-core footprint: `fill_row_depth * segment_bytes = 64 B`. It scales with no knob except `fill_row_depth`, which stays 1 by design.

## Precision Baseline

`tests/ttnn/unit_tests/operations/g6_fill/test_g6_fill_precision_baseline.py`: 4 production shapes x 3 values (`1.0`, `1/3`, the subnormal `1e-40`). Metrics use `comp_pcc`, `comp_allclose` and `comp_ulp`.

| Shape | Values | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | Max ULP | got/true ratio (median, p5, p95) | Bit-equal |
|-------|--------|-----|-------------|--------------|------------------|---------|----------------------------------|-----------|
| (185,) | 1.0, 1/3, 1e-40 | 1.0 | 0 | 0 | 0 | 0 | 1.0, 1.0, 1.0 | yes |
| (207,) | 1.0, 1/3, 1e-40 | 1.0 | 0 | 0 | 0 | 0 | 1.0, 1.0, 1.0 | yes |
| (256,) | 1.0, 1/3, 1e-40 | 1.0 | 0 | 0 | 0 | 0 | 1.0, 1.0, 1.0 | yes |
| (512,) | 1.0, 1/3, 1e-40 | 1.0 | 0 | 0 | 0 | 0 | 1.0, 1.0, 1.0 | yes |

**Assessment:** exact. There is no arithmetic and the 32-bit pattern is copied unchanged. The ratio spread is exactly 1.0, so there is no scale bug.

**Caveat:** PCC is degenerate for a constant tensor. `comp_pcc` falls back to allclose, and in the precompile collect pass, where no dispatch happens, it reported 0.999999 for a wrong output. Bit equality is the gate the test asserts, and it is the only meaningful one.

**Recommended tolerances:** exact (`helpers.TOLERANCES`: `("exact", None)` for float32 and int32). Defaults do not apply.

## Verifier CLI Summary

Graded run (`eval/eval_test_runner.sh`, defaults, so `EVAL_CAPTURE_L1=1`). Report: `verifier_results/verifier_report.json`.

- supported_pass: 15
- xfail_expected: 3 (int32 ROW_MAJOR `[1]`: `test_op`, `loose5` value bits, `loose10` refusals). All three are covered by Refinement 1.
- invalid_skipped: 0
- supported_fail: 1. `test_op_loose[loose6-512-…]` (trace), category `other`: `TT_FATAL mesh_workload.cpp:617 !is_capturing_trace — Cannot load new binaries during trace capture`. Harness-caused (H1). Not an OOM or precision cell.
- xpass_drift: 0
- xfail_wrong_mode: 0

Same suite with `EVAL_CAPTURE_L1=0` (`verifier_results/verifier_report_no_l1_capture.json`): **supported_pass 16, supported_fail 0, xfail_expected 3**.

Graded device kernel time, recorded by the harness for reference only and not a perf target (H3): 846–936 ns per single-call cell, 1 program, 64 B peak L1. The feature spec quotes stock ttnn at 1.34 µs for `[512]`.

## Harness Findings (escalations: these need eval-owner action, not op work)

**H1. Graded L1 capture makes every in-test trace capture fail.**
- `eval/l1_profiling.start_for_item` calls `device.disable_and_clear_program_cache()` and re-enables the cache only after the test body.
- With the cache disabled, every dispatch is a freshly built program. `mesh_workload.cpp:617` refuses to load new binaries while a trace is being captured.
- So any golden case that captures a trace inside the graded body fails for any op. g6_fill's `loose6` is the only such case in the eval tree today (`grep begin_trace_capture eval/golden_tests`), which is probably why this went unnoticed.
- Evidence: graded 15/16 supported; with `EVAL_CAPTURE_L1=0`, 16/16. The op's warm-up calls and cache key are correct (`test_new_value_reuses_cached_program`).
- Suggested fix: skip L1 capture for cases with `extras["trace"]`, or re-enable the program cache before the trace segment.
- I did not gate the cell out, add an EXCLUSION, or queue it: the op cannot change the outcome.

**H2. The zero-peak refusal forces a CB onto a design that does not need one.**
- `l1_profiling.capture_for_item` treats `peak_total <= 0` as an instrumentation failure. For a generator op whose source run can live in the common runtime args, 0 B of L1 is a legitimate result.
- The rule makes the faster `"common_args"` source unusable as the default: about 100 ns per call, roughly 17–23% of the implementer's profiled kernel time.
- Suggested fix: accept a zero peak when the graph capture shows the op's program was dispatched.

**H3. The perf-shape rule excludes every cell of this op, including the flagged one.**
- `eval/prompts/perf_rules.txt`: "Measure only cells of more than 32 tiles … this overrides any other instruction." Every g6_fill cell is at most 16 tiles (`[512]` read as `[1, 512]` is 16 tiles).
- `feature_spec.LOOSE_CASES[0]` flags `[512]` as the "mandatory target shape of every performance pass". `perf_refinement_prompt.txt` injects the same `perf_rules` ahead of its flagged-case rule.
- The two instructions conflict. Per the override, no perf refinement can be measured, so none is filed and the 2:1 cadence has no perf slot.
- If the eval owners exempt this op from the threshold, the levers in Recommendations are ready to file.

## Recommendations

- **Refinement 1 (int32 + ROW_MAJOR)** is the only TARGET gap. Keep `FILL_ROW_SOURCE = "cb"` there too (H2). If a `"common_args"` variant is offered for R2, its common-args length grows with `rm_chunk_bytes / 4` and needs its own cap against the kernel-config region (BB-04).
- Perf levers measured or identified but not filable under H3, all on `[512]` (the flagged case):
  1. Fill-row source `"common_args"`: -100 ns, measured; blocked by H2.
  2. Zero unique runtime args: derive `start_tile` / `core_w_tiles` from `get_relative_logical_x/y()` and common args, so dispatch writes one arg block per core range instead of one per core.
  3. Write issue path: `noc_async_write_one_packet_set_state` once per tile plus `_with_state` per segment. Only matters once `block_w_tiles >> 1`, and the grid lamp shows fewer cores loses, so expect nothing on production shapes.
  4. The grid-sync lamp is spent: the full spread is fastest (ledger audit table).

  These are T1 knobs from `master.md` (`width_split`, `noc_placement` already applied).
- No L1 or memory-pressure concerns: the footprint is 64 B, constant in `N`.
- No numerical-precision concerns: the op is exact by construction.
