# Verification Report: to_layout

Target: Quasar (`.claude/references/targets/quasar.md`) on ttsim (`.claude/references/backends/ttsim.md`).
Specification: `eval/specs/to_layout/spec.py` (sha256 82a27e071151…). Design: `op_design.md`. Ledger: `l1_ledger.md`.
Verified state: commit `c3fbaf55346` + this pass.

## Code Review

The implementation is clean and conforms to the design. The pass found no correctness or structural defect in the
supported surface. What was changed:

- **Added the R6 sharded-input test the op must carry itself** (`test_to_layout_plan.py::test_sharded_input_refused_by_name`).
  R6's check says "a sharded input is a test the op carries itself", and none existed. The test builds an L1
  HEIGHT-sharded bf16 tensor, asserts `eval.memory.memory_axes` tags it `{"L1", "height"}`, and asserts
  `to_layout` refuses it with `NotImplementedError` naming `sharding` before any kernel runs. Passes.
- **Added the precision baseline** (`test_to_layout_precision_baseline.py`), 4 shapes × 4 regimes, bit-exact (below).

Reviewed and found correct (no change needed):

- **Specification rules.** R1: every supported cell is bit-exact, through the bf16 datapath (unpack → 16-bit DEST → pack
  of a bf16 page), and the golden suite and the baseline both confirm it. R2: `validate()` refuses a block-float input with a
  ROW_MAJOR target (`UnsupportedAxisValue`), ahead of the SUPPORTED gate. bf8b is also outside SUPPORTED. R3: L1
  cells pass (60 of the 120). R4: every non-SUPPORTED cell raises a `SupportRefusal` (a `NotImplementedError`):
  1656 `xfail_expected`, 0 `xfail_wrong_mode`. R5: output allocated with
  `MemoryConfig(INTERLEAVED, input.memory_config().buffer_type)`, and `test_one_session_with_tt_metal_programs` asserts the
  buffer type for DRAM and L1 in every regime. R6: placement axes come from `memory_axes`, gated like any axis, and are
  now tested for a sharded input. Rejection X1: a non-`ttnn.Layout` `layout` raises `TypeError` (golden `test_rejection[X1]` passes).
- **One dispatch per invocation.** Every path, including the same-layout call (regime `identity_copy`), calls
  `ttnn.generic_op` exactly once, after `validate()`. No `ttnn.to_layout` / `tilize` / `untilize` wrapper.
- **Kernel API.** `void kernel_main()`, `api/...` include paths, `TensorAccessor` everywhere, Metal 2.0 `ProgramSpec` only
  (no `ProgramDescriptor`).
- **Ring protocols (target page, *Ring protocols*).** The explicit sides (tilize reader on `cb_row_block`, untilize writer on
  `cb_row_block_out`) list their ring in `disable_dfb_implicit_sync_for`, tag every transfer with trid 0 and barrier on it.
  The implicit sides use one tagged call per entry. Every kernel hands its tag back (`noc_async_{read,write}_set_trid(0, …)`),
  and the tilize reader resets the write tag too, because `async_write_zeros` reprograms it. The session test covers
  "after another program" and "before tt-metal's own `ttnn.to_layout`" in DRAM and L1.
- **Push count = wait count.** Every quantum is `band_tile_rows · bw` entries on rings of `BUFFER_DEPTH · BLOCK_HEIGHT_TILES · bw`
  entries. The reader's reserve/push, the helper's wait/pop/push and the writer's wait/pop match per block, and the
  implicit sides are one transfer per entry.
- **Helper usage.** Compute is one `compute_kernel_hw_startup` and one `compute_kernel_lib::tilize` / `untilize` call per
  kernel over all of the NEO's tile-rows (init / uninit once per kernel). The rejected dataflow helpers
  (`read_sticks_for_tilize`, `write_sticks_after_untilize`) are Metal 1 CB API with untagged transfers, which hang on Quasar
  after an implicit-sync program, so raw tagged NoC calls are the right realization. No multicast in the op, so no
  `McastArgs` question arises.
- **Broadcast efficiency.** Not applicable: no broadcast operand.

Latent issue found outside SUPPORTED (handed to a refinement, not fixed here because the cells are refused today):

- **Ragged-W stick transfers overrun the stick.** The stick-side reader and writer always move `chunk_bytes = bw·32·elem_size`.
  With `W % 32 != 0` that is more than the stick holds (W = 50 bf16: 100 B stick, 128 B transfer). A probe that bypassed
  `validate()` got bit-exact results for `w_non_aligned` shapes in DRAM, but only because the excess falls inside the
  64 B-aligned page slot. In L1 (16 B alignment) the untilize writer would write into the neighbouring page. The probe
  ran ranks 2/3 and ragged W through all four regimes and got 28/28 bit-exact (DRAM). Refinement 4 carries the
  exact-size-transfer fix as a hard requirement.

Advisories (GUIDANCE / soft):

- GUIDANCE says "a call that asks for the layout the input has needs no kernel". The op dispatches a DM-only copy instead,
  because the run's integrity rule requires exactly one dispatch per invocation (design, regime `identity_alias` rejected).
  This is consistent with the spec: DECISIONS 2026-11-19 asks only for "a tensor equal to the input".
- The shape dimensions (`image_height`, `image_height_tiles`, `tensor_w_tiles`, `total_tile_rows`, `n_col_blocks`,
  `chunk_bytes`) are compile-time args, so each distinct shape compiles its own kernel binaries. `bw` must stay compile-time
  (helper template, Quasar MOP width), but the rest could be runtime args to share binaries across shapes. This is a host /
  JIT-cache concern with no device-time effect and no failing cell, so it is a recommendation, not a refinement.
- The regime enum (`REGIME_TILIZE/UNTILIZE/COPY`) is restated in the Python module and in three kernels. Each kernel
  `static_assert`s the value is known, but a swapped mapping would not be caught. It is not a block knob. Keep the numbering
  in one place if a regime is ever added.

## Design Conformance

| Dimension | Design | Implementation | Verdict |
|-----------|--------|----------------|---------|
| Algorithm | tilize / untilize by unpacker + packer, DM-only page copy for same layout | same (`compute_kernel_lib::tilize/untilize`, `to_layout_copy_{reader,writer}.cpp`) | conforms |
| Pipeline / RISC ownership | reader → compute → writer per regime; identity reader → writer | same; rings by role name (`staged_input` / `staged_output`) | conforms |
| Work distribution | 2-D independent split, `(tile_row, col_block)` units flattened, `split_work_to_cores(row_wise=True)`, only clusters with work in `target_nodes` | same (`split_units`, `select_block_width_tiles`); 32 of 32 clusters engaged on 13 of 15 PHASE0 shapes (worked table re-verified by `test_select_block_width_tiles_worked_table`) | conforms: fills the grid |
| Both dataflow halves batched | explicit side: one reserve/wait, ≤ 32 transfers, one barrier, one push/pop per block; implicit side: one tagged call per entry, no barrier | same | conforms |
| Inter-core communication | none | none | conforms |
| Thread plan | (1, 1, 1) from one constant `THREAD_PLAN`, stated as data-movement bound with the reason | `THREAD_PLAN = (1, 1, 1)` → `KernelSpec(num_threads=…)` and CT args `reader/compute/writer_threads`; kernels stride by them from `get_my_thread_id()`; `test_thread_plan_on_every_kernel` reads it off every built `ProgramSpec`; session tests present and passing | conforms. ttsim has no timing model, so the alternatives the design named ((2,2,2) BLOCKED) were not measured here. The README's timed figures for explicit sync motivate Refinement 5 |
| Blocking-model fidelity | knobs `BLOCK_HEIGHT_TILES`, `select_block_width_tiles()`, `BUFFER_DEPTH`, `COPY_BLOCK_PAGES` | each defined once on the host; ring capacities, `chunk_bytes`, `n_col_blocks`, the compute kernel's `num_tile_rows` all derived from them; kernels take them as CT args. No CB scales with an op dimension. No half-turned split (the per-core loop walks whole `bw`-tile blocks) | conforms (DRY holds) |
| Expression | block operations at block boundaries | reader / writer acquire, sync and retire per block (explicit) or per entry by protocol (implicit); compute one helper call over all blocks | conforms |

## Registry Conformance

- `INPUT_TAGGERS = AXES` imported from `eval.specs.to_layout.spec`, with no op-local axis function. `SUPPORTED` equals `PHASE0`
  (lint `--phase0` passes). `EXCLUSIONS = []`. `validate()` runs X1 type check → placement axes via `memory_axes` → spec taggers →
  R2 → per-axis SUPPORTED → EXCLUSIONS, raising `UnsupportedAxisValue` / `ExcludedCell`. The entry point calls
  `validate()` first. The op file declares no `INVALID`.
- No SUPPORTED auto-fixes: `xpass_drift = 0`.
- **INVALID audit (specification).** One entry, `{dtype: bfloat8_b, layout: ROW_MAJOR}`: single tensor (the input), structurally
  impossible (a block-float format has no row-major form), the canonical bf8b + ROW_MAJOR activation entry. No cross-tensor
  coupling, no "not yet" entry. Not a norm op, so no weight canonicalization. Passes.
- `python3 -m eval.validate_contract to_layout --phase0 --text`: PASSED, 0 errors, 5 UNRESOLVED open items (spec-side,
  informational: `governs`, `special_values`, `freedoms`, `PERFORMANCE.bar/measure`). No specification defect found.

## L1 Ledger Audit

- **Ledger currency.** Each program declares one regime's rings, and all five rows match the code. Stream rings:
  `entry_size = tile_bytes`, `num_entries = BUFFER_DEPTH · BLOCK_HEIGHT_TILES · bw · C`, format = input dtype. `cb_pages`:
  `entry_size = buffer_page_size()`, `num_entries = BUFFER_DEPTH · COPY_BLOCK_PAGES`, no format. Ring names match the design.
- **Capacity vs live set.** Each stream ring holds one `bw`-tile band live (the unpack-tilize MOP reads all 32 rows of
  the band, and pack-untilize writes rows across all `bw` tiles), plus one band for double buffering. That is justified, with no
  over-sizing. Axis accounting is correct: rings span `tile_col` / `row_in_tile` / `col_in_tile` and stream over `tile_row`.
  `cb_pages` is a streaming window over `page`, not spanning it. No collapsed extent.
- **Page format vs DEST width.** bf16 pages with `enable_32_bit_dest=False`: widths match. `cb_pages` has no format (bytes never pass
  DEST). Refinement 2 must widen both DEST and the page together.
- **Disjoint lifetimes.** None: every regime's two rings are concurrently live (pipelined), and the helpers forbid in-place
  (`static_assert(input_dfb != output_dfb)`). Justified in the ledger.
- **Bounds / closed form.** `bw ≤ bw_max` (ring cap 1,048,560 B, budget `get_max_worker_l1_unreserved_size() − 1 MiB`),
  `BLOCK_HEIGHT_TILES = 1`, `BUFFER_DEPTH = 2`, `COPY_BLOCK_PAGES = 4`, page size ≤ 16 KiB on supported shapes. No op
  dimension in a capacity expression.
- **Block-size defaults.** Interleaved default held: units spread over the full grid first, then the coarsest `bw` that still
  fills it (step 4), floored at 256 B stick transfers. The `BLOCK_HEIGHT_TILES = 1` departure from "coarsest" is justified in the
  design: 1–2 units per cluster on PHASE0 shapes, and compute fixed cost is paid once per kernel.
- **Per-core footprint.** Stream: `2 · BUFFER_DEPTH · BLOCK_HEIGHT_TILES · bw · tile_bytes` = `8192·bw` B for bf16 (PHASE0 max
  128 KiB at `bw = 16`), scaling with `bw`, `BUFFER_DEPTH`, `BLOCK_HEIGHT_TILES`. Identity: `BUFFER_DEPTH · COPY_BLOCK_PAGES ·
  page_bytes` (max 128 KiB).
- **Data-movement budget.** Present and consistent with the code: tilize `S_rm + S_tile` (padding rows zeroed in L1, never read),
  untilize `S_tile + S_rm` (padding rows of each image's last tile-row read as part of whole tile pages, which is structural at
  page granularity and is Refinement 3 lever 2), identity `2 × size`, zero cross-core traffic. The cheapest-traffic split is the
  implemented one. No deferred regime row is missing.
- **Filing.** No ledger finding needed a fix. The identity window, which counts pages rather than bytes, is folded into
  Refinement 6. The ragged-W page alignment of `cb_pages` is folded into Refinement 4.

## Precision Baseline

`tests/ttnn/unit_tests/operations/to_layout/test_to_layout_precision_baseline.py`: bf16, DRAM, seed 1234, standard-normal inputs.
All four regimes on each shape gave identical figures. Results are written to `generated/to_layout_precision_baseline/*.json`.

| Shape | Regimes | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | Max ULP | got/true ratio (median, p5–p95) | Bit-equal |
|-------|---------|-----|-------------|--------------|------------------|---------|----------------------------------|-----------|
| (1,1,32,64) | tilize, untilize, copy_tile, copy_rm | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 1.0, 1.0–1.0 | yes |
| (1,1,400,512) | same | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 1.0, 1.0–1.0 | yes |
| (1,1,1600,256) | same | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 1.0, 1.0–1.0 | yes |
| (2,4,128,512) | same | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 1.0, 1.0–1.0 | yes |

**Assessment**: bit-exact on every shape and regime: no rounding, no scale error (ratio exactly 1). This is the spec's
`exact` requirement (R1, NUMERIC_RESOLVED movement class).
**Recommended tolerances**: exact (every cell's resolved entry is `exact`). No legacy tolerance applies.

## Verifier CLI Summary

Golden run: `eval/eval_test_runner.sh eval/golden_tests/to_layout/ /tmp/to_layout_golden` (8 sim workers, forked), 2074 tests.
Report: `/tmp/to_layout_golden/verifier_report.json` (copied to `pipeline_logs/verifier_report.json`).

- supported_pass: 120
- xfail_expected: 1656
- invalid_skipped: 296 (148 INVALID cells plus the bf8b cells skipped as unbuildable on Quasar)
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0
- supported_marked_xfail / invalid_unexpected: 0 / 0
- no axes sidecar: 2 (`test_rejection[X1]` passed; `test_op_loose[NOTSET]` skipped because there are no loose cases)

Queue check (`--queue op_requirements.md`): 9 values in TARGET − SUPPORTED, 0 queue gaps.

Other suites: acceptance + plan tests 48/48 at entry, and 40/40 for the plan + baseline files after this pass's additions.

## Recommendations

- **Queue order** (op_requirements.md): R1 narrow integers (hardest: no known bit-exact unpack path; uint16 is not a ring
  format and its values 1–127 are bf16 denormals), R2 32-bit exact path, R3 perf on the YOLOv8 calls (bank stagger + untilize
  half-tile reads), R4 ranks 2/3 + ragged W (includes the transfer-overrun fix), R5 perf on the explicit-sync side's thread plan
  (T3), R6 perf on the identity window (T1). bfloat8_b is omitted: Quasar has no path for it.
- **Perf on this backend is structural.** ttsim reports no clock. Every perf refinement names a count and its before/after
  value, and its time gain stays unconfirmed until `ttsim-perf` or hardware measures it. The design's lamps that need
  timing to decide (occupancy vs stick-transfer size at `bw = 4` / 256 B on `[1,1,400,512]`, block height) are not queued, because
  no structural count settles them. Measure them on `ttsim-perf` when a run has it.
- **Load balance on the YOLO shapes**: 52 and 50 units over 32 clusters give a makespan of 1.23× and 1.28× the mean.
  A better balance needs sub-floor stick transfers (`bw` ≤ 2), so it is noted here, not queued.
- **Special values** (NaN / Inf, bf16 denormals) are UNRESOLVED in the spec and no golden input carries one. The bf16 SrcA path may
  canonicalize denormals and NaN payloads. If the spec later requires them moved unchanged, the bf16 path needs the same
  bit-exact treatment Refinement 1 builds for uint16.
