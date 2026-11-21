# Verification Report: plus_one

Target: Quasar (`quasar`) on the ttsim simulator (untimed, slow dispatch). Specification: `eval/specs/plus_one/spec.py` (lint: PASSED, 0 errors, 4 UNRESOLVED open items that are the spec author's — `DEFINITION.tt_regen_spec.{row,governs}`, `PERFORMANCE.{bar,measure}` — none blocks the op). Design: `op_design.md`, Regime A `segment_split_in_place`.

## Code Review

What the op is: one `ttnn.generic_op` dispatch of a Metal 2.0 `ProgramSpec` whose only kernel is a data-movement `worker` with `num_threads = WORKER_THREADS = 4` per engaged cluster. Each thread reads a block of its units (row segments) into its own 64-B-line-aligned region of one scratchpad (tagged trid-0 reads, one barrier per block), increments the 32-bit words as `uint32_t` inside one `scoped_lock` (L2 invalidate on acquire, flush on release), and writes the same bytes back to the same page offsets (tagged writes, one barrier per block). `io_tensors = [input_tensor]`, nothing allocated, the caller's object returned (R3).

Fixed in place:

1. **DRY, host** (`plus_one_program_artifacts.py`): the common-runtime-arg schema restated the seven geometry names in a list beside the dict that carried their values; a new arg added to one and not the other would only fail at dispatch. The schema is now `list(common_values)` — one definition.
2. **DRY, kernel** (`kernels/plus_one_worker.cpp`): the unit → (row, offset, length) mapping (incl. the ragged last segment) was written out three times, in `load_block`, `increment_block` and `store_block`. It is now one `segment_of()` lambda returning a `Segment`; the three phases call it. A drift between the read extent and the write extent of a unit would be a silent partial write, so this is a correctness-relevant DRY fix, not cosmetics.

Both fixes re-verified: unit tests 47/47 (incl. the new baseline), golden 28/28 supported cells, no change in any verifier category.

Checked and found correct (no change):

- **Rules.** R1: all arithmetic on `uint32_t` (`x + increment` wraps; no float format anywhere — no compute kernel, scratchpad has no data format). R2: the flag test is `x < 0x7FFFFFFF` on the raw word, which equals "int32 view in [0, INT32_MAX)" for both dtypes. R3: in place, returns `input_tensor`; golden `HOOKS["check"]` passes in every supported cell. R4: axes from `eval.memory.memory_axes`. R5: `UnsupportedAxisValue` / `ExcludedCell` subclass `NotImplementedError`; every out-of-phase cell refused (148 `xfail_expected`, 0 `xfail_wrong_mode`). No rule has a `structure` part; `GUIDANCE` (DM-hart integer add, one owner per page, padding untouched) is followed: transfers are exactly `n·4` bytes from `off·4`, units partition the elements exactly.
- **Kernel hygiene.** `void kernel_main()`, `api/dataflow/...` include paths, `TensorAccessor` (no `InterleavedAddrGen`), no DFB so no push/pop accounting to balance. Every transfer is tagged (user trid 0) and the tag is handed back at the end (target page, *Ring protocols*); the session test (`untilize → plus_one → untilize → plus_one → untilize`, one device session, x + 2 expected) passes.
- **Helpers.** No kernel-library helper fits: the eltwise helpers are NEO-side over DFB tiles, `local_copy`/`dfb` helpers need a DFB (a DM self-looping DFB is refused on Gen2), `addr_to_l1_ptr` would bypass the scoped-lock cache maintenance the design requires. The design's rejection table is accurate; raw NoC API is the right mechanism. No multicast exists (no shared operand).

## Registry Conformance

- `INPUT_TAGGERS = AXES`, imported from the specification; no op-local axis function.
- `SUPPORTED` covers every axis the op gates on (`dtype`, `layout`, `memory_config`, `sharding`, `skip_negative_entries`, `rank`) and equals `PHASE0` (lint `--phase0` passes).
- `EXCLUSIONS = []`; `validate()` checks SUPPORTED per axis then EXCLUSIONS, raising the `_op_contract` refusals; it is the first statement of the dispatch path (`_plus_one`, which the public `plus_one` calls).
- The op file declares no `INVALID`.
- No auto-fixes to SUPPORTED: no XPASS evidence (validate refuses every out-of-phase cell, so none can XPASS). A verifier probe shows the 148 refused cells already compute exactly through the builder — that is Refinement 1's evidence, not drift.
- **INVALID audit (specification).** `INVALID = []`. Correct: TARGET has one layout (ROW_MAJOR) and only integer dtypes, so the canonical `bfloat8_b + ROW_MAJOR` cell cannot arise and needs no entry; one tensor, so no cross-tensor coupling; not norm-like. Nothing to report upstream.

## Design Conformance

- **Algorithm / pipeline / RISC ownership**: as designed (Regime A, one DM kernel, no compute kernel, no DFB, scratchpad).
- **Work distribution**: `split_work_to_cores(..., row_wise=True)` over `num_cores = ceil(num_workers / WORKER_THREADS)` clusters; threads stride units by the CT `WORKER_THREADS`. Tiny tensors (≤ 1 KiB) use one worker by the design's `MIN_WORKER_BYTES` floor — a justified grid-synchronization choice for this op's purpose (positions, ≤ 100 elements), not an under-filled grid. `(4096,)` runs on 4 clusters × 4 threads (16 units); `(1,1,2,8192)` would run on 16 clusters (probe).
- **Thread plan** (read off the host program): one kernel, `num_threads = WORKER_THREADS` (one named constant, passed as a compile-time arg; the kernel strides by it, never by `get_num_threads()`), C = 0. The target page's default for a data-movement-bound op is (1,1,1); the design departs to 4 threads **with a stated reason** (the increment runs on the DM harts, so per-word ALU work grows with the share while transfer time barely does). The plan's required tests are present and pass: every thread ran (`probe_tag_threads`, thread t adds 1 + t, owner recomputed from the host split, all four ids present), one device session (above). No buffers between kernels, so no per-thread ring sizing or protocol to check. ttsim has no timing model, so the plan was not measured against the named alternatives (`WORKER_THREADS ∈ {1, 6}`); `pipeline_logs/model_cycles.jsonl` does not exist for this run.
- **Blocking-model fidelity**: every knob (`WORKER_THREADS`, `MIN_WORKER_BYTES`, `SCRATCH_DEPTH`, `L1_RESERVE_BYTES`, `CACHE_LINE_BYTES`) is one host constant; `seg_elems`, `block_units`, `slot_stride_bytes`, `region_stride_bytes`, scratch size are derived once in `split_plan` and passed as args; no CB/scratch dimension scales with an op dimension (bounded by `max_block_bytes`). Expression: each block is acquired with `n_units` back-to-back reads and **one** barrier, incremented under **one** lock, stored with **one** write barrier — block-scoped on all three halves, no per-unit completion boundary. Not half-turned: a thread's whole share is one block in every INPUTS shape.
- **Not applicable on this part**: the design's "writes on the other NoC" perf lamp — Quasar has `NUM_NOCS 1` (`tt_metal/hw/inc/internal/tt-2xx/quasar/noc/noc_parameters.h:32`).

## L1 Ledger Audit

- **Currency**: one L1 allocation (scratchpad `scratch_units`), one row; its size expression `WORKER_THREADS·align_up(SCRATCH_DEPTH·block_units·slot_stride_bytes, 64) + 64` matches `split_plan.scratch_bytes` exactly.
- **Capacity vs live set**: at `SCRATCH_DEPTH = 1` (the only value run) capacity = live set + line/alignment round-up (mechanism-justified: L2 line exclusivity). **Finding**: at `SCRATCH_DEPTH > 1` capacity grows but the serial kernel never holds a second block live (the depth knob buys L1, not overlap). No INPUTS shape gives a thread more than one block, so nothing is wasted today; recorded in the ledger (verifier note added), not queued (see Recommendations). Axis accounting (`row` spans `block_units`, `col` spans `seg_elems`) is truthful: both extents are simultaneously live during `increment_block`.
- **Page format vs DEST**: n/a — no compute kernel, no DEST; the scratchpad carries no format (uint32 is not a buffer format on Quasar).
- **Disjoint lifetimes**: n/a — a single buffer; the per-thread non-sharing is justified (concurrent lifetimes, line exclusivity).
- **Bounds**: every symbol bounded in the ledger's table; `W`, `rows` never enter a capacity expression.
- **Data-movement budget**: present and consistent with the code — DRAM crossings 2 per element (read once, written once), 0 cross-core bytes, transfers `2·rows·segs_per_row`. Cheapest-traffic split is the implemented one (row-first; segments only where rows under-fill). The cheaper-than-DRAM-minimum L1 bank-local regime (E) is a `deferred` row with a positive reason (L1 not in PHASE0); it is now Refinement 2.
- **Block-size defaults**: held — the interleaved default (spread units over the wanted workers, then the coarsest block that fits: the thread's whole share) is what `split_plan` computes.
- **Per-core footprint**: `L1 = WORKER_THREADS·align_up(SCRATCH_DEPTH·block_units·align_up(seg_elems·4, l1_align), 64) + 64`; linear in `WORKER_THREADS`, `SCRATCH_DEPTH`, `block_units` (row), `seg_elems` (col); ≤ budget by construction. Phase 0 worst case: 4,160 B per cluster (`(4096,)`).

## Precision Baseline

`tests/ttnn/unit_tests/operations/plus_one/test_plus_one_precision_baseline.py` — inputs over the whole 32-bit range with the rule edge values (wrap points, the flag's window ends) at the front; 16 cells = 4 shapes × {int32, uint32} × flag. ULP = integer distance modulo 2^32.

| Shape | Cells | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | Max ULP | got/true ratio (median, p5, p95) |
|-------|-------|-----|-------------|--------------|------------------|---------|-----------------------------------|
| (1,) | 4 | 1.000000 | 0 | 0 | 0 | 0 | 1.0, 1.0, 1.0 |
| (17,) | 4 | 1.000000 | 0 | 0 | 0 | 0 | 1.0, 1.0, 1.0 |
| (100,) | 4 | 1.000000 | 0 | 0 | 0 | 0 | 1.0, 1.0, 1.0 |
| (4096,) | 4 | 1.000000 | 0 | 0 | 0 | 0 | 1.0, 1.0, 1.0 |

**Assessment**: bit-exact in every cell, both dtypes, both flag values, including INT32_MAX → INT32_MIN, 2^32 − 1 → 0, and the flag leaving INT32_MAX / negatives / uint32 ≥ 2^31 − 1 untouched. No scale signature (ratio spread is exactly 1). Additionally the verifier probe ran the 148 out-of-phase cells through the builder: all bit-exact.
**Recommended tolerances**: exact (every cell resolves to `exact` in `NUMERIC_RESOLVED`).

## Verifier CLI Summary

Results dir: `ttnn/ttnn/operations/plus_one/golden_results/` (`verifier_report.json`). 177 tests (176 cells + 1 empty `LOOSE_CASES` placeholder).

- supported_pass: 28
- xfail_expected: 148 (L1 × rank 1: 28; ranks 2–4 × {DRAM, L1}: 120 — all covered by Refinement 1)
- invalid_skipped: 0 (INVALID is empty)
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0
- no_axes_found: 1 (`test_op_loose[NOTSET]`, skipped: `LOOSE_CASES` is empty)
- Queue coverage (`--queue op_requirements.md`): 4 values in TARGET − SUPPORTED, 0 gaps.

## Recommendations

- **Queue shape.** `TARGET − SUPPORTED` is just `memory_config: L1` and `rank: 2, 3, 4`, both knob-turns that already compute exactly; they are one bundled generality refinement (Refinement 1), below the usual 3–5 because there is nothing else to unlock. Generality is then exhausted, so the queue continues with perf: Refinement 2 (L1 bank-local, structural gate: NoC bytes per call → 0), anchored on Refinement 1 landing L1 and ranks 2–4.
- **Perf levers recorded, not queued (no structural count to move, or no target shape):**
  - `SCRATCH_DEPTH = 2` overlap (issue block k+1's reads before retiring block k, à la `trid_double_issue`): only acts when a thread has > 1 block, i.e. a share above `max_block_bytes` ≈ 1 MiB; the largest INPUTS tensor is 64 KiB. Revisit if a large-tensor loose case is added. Until then depth > 1 is a dead knob (ledger note).
  - `MIN_WORKER_BYTES`, `WORKER_THREADS` (1 / 4 / 6), threads-first vs cores-first dealing, per-thread trids: all trade occupancy against per-worker fixed cost, which only a timed backend can rank. Re-open on ttsim-perf or hardware with the design's perf-lamp table as the sweep list.
  - Writes on a second NoC: not available (Quasar `NUM_NOCS 1`).
- **Rank-1 DRAM tensors** are one page in one DRAM bank; every segment of a split `(4096,)` hits the same bank. Inherent in the placement (design's traffic ranking); no lever.
- **Specification open items** (`UNRESOLVED`: `tt_regen_spec.row/governs`, `PERFORMANCE.bar/measure`) are for the spec author; until `PERFORMANCE` names a case and a bar there is no perf-flagged target, which is why Refinement 2's region was free-selected.
