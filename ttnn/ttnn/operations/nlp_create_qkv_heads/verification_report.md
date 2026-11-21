# Verification Report: nlp_create_qkv_heads

Target: Quasar (`quasar`), backend: ttsim simulator (no clock). Specification:
`eval/specs/nlp_create_qkv_heads/spec.py` (sha256 9202df1df2e0). Phase 0 regime: `tile_relocate`.

## Code Review

**Fixed**
- `nlp_create_qkv_heads.py::output_shapes` re-derived the three result shapes that the specification's
  `head_shapes` already defines. It now calls `head_shapes`, so the result shapes have a single definition
  (and the transposed-k shape Refinement 2 needs is already right). Tests re-run green.

**Reviewed, conformant, nothing to fix**
- Kernels: `void kernel_main()`, `api/dataflow/...` includes, `TensorAccessor` on every tensor, Metal 2.0
  `ttnn.ProgramSpec` (no Metal 1 descriptor), one `ttnn.generic_op` dispatch per call (task rule).
- Ring protocol: one DM→DM STRIDED ring, implicit sync on both sides, one tagged NoC call per entry, no
  reserve/push/barrier; both kernels hand their sticky transaction tag back as their last statement
  (target page *Ring protocols*). No CB sync mismatch is possible (no explicit credits).
- No compute kernel, so no compute helpers apply; the design's "helpers considered and rejected" table is
  correct (no kernel-library helper does a tagged accessor↔ring transfer). No multicast / semaphores, so no
  `McastArgs` to substitute. No broadcast.
- Writer walk: start state decomposed on the host, advanced on device by carries only (no division).

**Deferred to refinements**: none from code review. The design-level item below (k-transpose mechanism) is a
note in Refinement 2.

## Specification Rules

| Rule | Applies in Phase 0 | Status |
|------|--------------------|--------|
| R1 exact movement | yes | Met: the ring's format is the input dtype, no DEST, no conversion. Precision baseline bit-exact. |
| R2 list of three, shapes/dtype/layout | yes | Met (hook checks every tensor; golden green). |
| R3 transposed k | no (`False` only) | Refused through SUPPORTED. Refinement 2. |
| R4 `num_kv_heads=None` | yes | Met: `((1,1,64,4608), 12, None)` passes. |
| R5 output placement | yes (DRAM) | Met: each result allocated on its own with `allocate_tensor_on_device`, buffer from `memory_config`, else the input's (`None` path already coded; probe-tested for L1). |
| R6 `memory_config` read by `canonical_memory` | yes | Met (X5 raises `TypeError`; a sharded config reads `"<buffer>/<sharding>"`, outside SUPPORTED). |
| R7 DRAM or L1 input | partially (DRAM) | L1 input works (probe_003) but is not claimed yet: Refinement 4. |
| R8 `memory_axes(input)` gating | yes | Met: `axes_of` derives `memory_config`/`sharding` from `eval.memory.memory_axes`. |
| R9 row-major refused by name | yes | Met: `layout` axis outside SUPPORTED → `UnsupportedAxisValue` naming it (X6 passes). |
| R10 `NotImplementedError` outside support | yes | Met: `UnsupportedAxisValue`/`ExcludedCell` subclass `NotImplementedError`; 423 xfail_expected, 0 wrong-mode. |

No rule has a `structure` part. **GUIDANCE**: all applicable advice is followed (no compute in Phase 0, whole
tiles moved, `allocate_tensor_on_device` for the three results, a work unit finer than a tile-row when rows are
fewer than cores). No advisories.

## Registry Conformance
- `INPUT_TAGGERS = AXES`, imported from the specification; no op-local tagger.
- `SUPPORTED` covers every axis the op gates on (dtype, layout, memory_config, sharding, transpose_k_heads,
  output_memory, alignment, batch) and equals `PHASE0` (lint `--phase0`: PASSED, 0 errors).
- `EXCLUSIONS = []`. `validate()`: structural rejections X1–X4 (`ValueError`) first, then SUPPORTED per
  axis, then EXCLUSIONS; called first by the entry point. The op file declares no `INVALID`.
- No auto-fixes to SUPPORTED (no XPASS evidence: validate() refuses every non-PHASE0 cell, as it must).
- **INVALID audit (specification)**: `INVALID = []`. Correct: no cross-tensor coupling exists (one input),
  `layout` TARGET is TILE only so the canonical bf8b+ROW_MAJOR cell cannot arise, bfloat8_b is not in TARGET
  (and unbuildable on Quasar), and the op is not norm-like. No specification defect.

## Design Conformance
- **Algorithm**: tile permutation, bytes untouched — as designed.
- **Topology / RISC ownership**: reader → `input_tiles` ring → writer, no compute kernel — as designed.
- **Work distribution**: `split_work_to_cores(grid, num_pages, row_wise=True)` over the folded input-page axis;
  `min(num_pages, 32)` clusters engaged, shares differ by ≤ 1 page (the traced calls use all 32). Both halves
  stream: one tagged transfer per page each side, no per-page barrier; ring depth 8 overlaps reads with writes.
- **Thread plan**: (R, C, W) = (1, –, 1) from one host constant `THREAD_PLAN`, passed as compile-time args;
  the design states the op DM-bound with its reason (target page *Choosing the plan*),
  so one thread of each kind is the sanctioned plan. Kernels stride by
  their thread counts; `test_knobs_exact` runs (2,2), (1,2), (2,1), (2,4), (3,3) exactly (a thread that did
  not run leaves pages unwritten). Device-session tests (op after quasar softmax and after itself;
  `ttnn.to_layout` after the op) present and passing. No timed backend in this run, so the plan alternatives
  were not measured (the backend page gives no clock).
- **Blocking-model fidelity**: block knobs `BLOCKS_PER_CORE`, `RING_DEPTH_TILES`, `THREAD_PLAN` are host
  constants, each defined once, overridable per call, every dependent quantity (ring entries, block pages,
  loop strides) derived from them. No CB scales with an op dimension. The per-core loop moves one page per
  NoC transaction, which is the implicit-sync transfer quantum, not a half-turned knob: there is no
  per-unit handshake, and blocking only changes the outer trip count.
  Expression check: no planned extent > 1 is retired per unit.
- **Design note (deferred regime)**: the `tile_relocate_k_transpose` row plans a compute `transpose_tile`
  through DEST with `fp32_dest_acc_en` for float32. Quasar's
  unpack-to-DEST path cannot transpose (only SrcA can), and nothing shows the SrcA path bit-exact for float32
  (R1). Recorded as a mechanism warning in Refinement 2; not a Phase 0 defect.

## L1 Ledger Audit
- **Currency**: one row, `cb_input_tiles`; code allocates `DataflowBufferSpec(entry_size=page_bytes,
  num_entries=ring_entries(threads, depth))` with `ring_entries = align_up(RING_DEPTH_TILES, 4·max(R, W))` —
  matches the ledger.
- **Capacity vs live set**: capacity = the in-flight window (2 txn-id slices of 4); spans no block axis,
  streams over all of them — correct in both directions.
- **Page format vs DEST**: no DEST in the program; a float32 page (future) is a dataflow-only payload, which
  R1 requires. No finding.
- **Disjoint lifetimes**: single buffer; nothing to share.
- **Bounds / closed form**: footprint `ring_entries · page_bytes`, bounded by constants (≤ 196,608 B at the
  lamp maximum against 3,963,328 B); no op dimension appears.
- **Data-movement budget**: present and consistent with the code: each input page read once, each output
  page written once, 0 cross-core bytes; the cheapest-traffic split is the implemented one.
- **Block-size defaults**: held — interleaved split over the full grid, then the coarsest block
  (`BLOCKS_PER_CORE = 1`).
- **Per-core footprint**: `ring_entries(RING_DEPTH_TILES, THREAD_PLAN) · page_bytes` = 16,384 B (bf16, Phase 0).
  Scales with the ring-depth knob, the thread plan (alignment) and dtype only.

## Precision Baseline

`tests/ttnn/unit_tests/operations/nlp_create_qkv_heads/test_nlp_create_qkv_heads_precision_baseline.py`,
bf16, DRAM → DRAM, standard-normal input, seed 42. Identical for q, k and v of every shape:

| Shape (heads, kv heads) | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | Max ULP (bf16) | got/true ratio median [p5, p95] |
|-------|-----|-------------|--------------|------------------|-----|----|
| (1,1,32,96) (1, 1) | 1.0 | 0 | 0 | 0 | 0 | 1 [1, 1] |
| (1,1,128,3072) (16, 16) | 1.0 | 0 | 0 | 0 | 0 | 1 [1, 1] |
| (1,1,128,6144) (32, 8) | 1.0 | 0 | 0 | 0 | 0 | 1 [1, 1] |
| (1,1,4096,96) (1, 1) | 1.0 | 0 | 0 | 0 | 0 | 1 [1, 1] |

**Assessment**: bit-exact, as R1 requires; no scale signature (ratio exactly 1 everywhere).
**Recommended tolerances**: exact (every cell resolves to `exact` in `NUMERIC_RESOLVED`).

## Verifier CLI Summary
- supported_pass: 9
- xfail_expected: 423
- invalid_skipped: 0
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0
- no_axes_found: 7 (6 rejection tests X1–X6, all passing; 1 empty `test_op_loose` parameter set — `LOOSE_CASES = []`)
- queue_gaps: 0 (all 8 values of TARGET − SUPPORTED are named in `op_requirements.md`)

Results: `/tmp/nlpqkv_golden2/` (copy of the JSON: `verifier_report.json` in this directory).
Op tests: 44 passed (acceptance 4, implementer plan/knob/session 36, precision baseline 4).

## Queue rationale (TARGET − SUPPORTED)

| Axis value | Refinement | Class |
|-----------|------------|-------|
| alignment `w_non_aligned` | 1 | scheme-change (`subtile_assemble`) — stands alone |
| transpose_k_heads `True` | 2 | new per-tile stage on k — stands alone |
| dtype float32; memory_config L1; output_memory None, L1; alignment h_non_aligned; batch batched | 4 | knob-turns, one bundle; probe_003 shows `tile_relocate` already exact on all of them |

Perf slots (2:1 cadence): Refinement 3 (block/thread co-tune of the two new paths) and Refinement 5 (DRAM bank
stagger on the traced calls). No `LOOSE_CASES` entry is perf-flagged, so the regions are free-selected; the
traced calls are already supported, so no unlock had to precede the first perf slot, and the generality
refinements are ordered hardest first.

## Recommendations
- **Perf headroom of `tile_relocate` is small, and most of it cannot be measured here.** It is at the
  structural minimum (1 DRAM read + 1 write per page, 1 NoC transaction per page, 0 cross-core bytes, full
  grid, implicit sync at plan (1, –, 1)). The design's
  remaining lamps — ring depth, thread plan (2, –, 2), engaging fewer clusters on tiny calls, output-order
  walk — have no structural count on ttsim and are left for a `ttsim-perf` or hardware run. Only the bank
  stagger has a count, hence Refinement 5.
- Refinement 4 is cheap for `tile_relocate` (SUPPORTED + tests); its real cost is extending Refinements 1/2
  to float32 exactly. Keep element sizes derived from the dtype in those paths.
- No L1 pressure: Phase 0 uses 0.4 % of the per-core budget; Refinement 1 adds resident tile-rows
  (`Wt · page_bytes`, ≤ 512 KiB for an 8192-wide fp32 row) and must keep them bounded in the ledger.
