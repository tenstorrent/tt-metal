# Verification Report: sigmoid

Target: Quasar (`quasar`, 8 × 4 = 32 clusters) on the ttsim simulator. Specification: `eval/specs/sigmoid/spec.py`.
Golden results: `/tmp/sigmoid_verifier_golden/`; its `verifier_report.json` is copied beside this file.

## Code Review

**Fixed**

- **`dst_full_sync_en` was silently dropped** (`sigmoid_program_artifacts.py`). The caller's
  `ComputeConfigDescriptor.dst_full_sync_en` never reached the hardware config; DEST always ran half-sync and
  `block_tiles` assumed it. This contradicts the spec's GUIDANCE ("pass the config as-is") — a config field
  ignored without a refusal. Now mapped as tt-metal does (`compute_kernel_config.cpp`):
  `double_buffer_dest = not dst_full_sync_en`, and the one block knob follows both DEST fields:
  `DEST_TILES[(fp32_dest_acc_en, dst_full_sync_en)]` = 4 / 8 / 8 / 16 (= `get_dest_limit()` in
  `kernel_lib/dest_helpers.hpp`). Covered by `test_config_pass_through[full_sync]` (block_tiles = 8 build,
  0 ULP). `op_design.md` (config table, `sync_block` row) and `l1_ledger.md` (`block_tiles` bound, worst-case
  footprint) updated to match.

**Reviewed, no change needed**

- *Specification rules*: R1 holds (float32 is outside SUPPORTED; becomes a permanent EXCLUSIONS entry with
  Refinement 2, written into its goal). R2: `default_compute_kernel_config()` is the only place the default is
  defined, `None` resolves through it, exported from `__init__`. R3: `validate()` reads neither
  `math_fidelity` nor `math_approx_mode`; both are passed through (LoFi + approx run, 0 ULP). R4/R6: DRAM and L1
  inputs read through `TensorAccessor`; output allocated from `input_tensor.spec`, so it inherits the input's
  memory config, interleaved. R5/R7: axes from `eval.memory.memory_axes`, refusal through `_op_contract`
  (`NotImplementedError` subclasses). No `structure` rules in the spec. `REJECTIONS` empty.
- *Helper usage*: compute is one `eltwise_chain(CopyTile → SigmoidAccurate → PackTile)` per NEO, PerBlockSize
  wait/pop/reserve/push in DEST-sized blocks; init / reconfig hoisted by the chain. The op-local chain element
  (raw SFPI body around `_sfpu_sigmoid_<DST_ACCUM_MODE>`) is a justified deviation: the library `Sigmoid<>` is
  the LUT exp + LUT reciprocal (~7-bit mantissa) path on Quasar and cannot hold the ULP gate; `add_unary_tile`
  is compiled out on Quasar so no helper-only composition exists. Mechanism, not a finding.
- *Correctness*: `void kernel_main()`, `api/...` includes, `TensorAccessor`, no deprecated APIs. Push = wait
  per ring (reader credits `neo_tiles` per NEO slice; NEO pops/pushes `neo_tiles`; writer drains them). Tag
  hand-back as last statement of reader and writer; `test_one_device_session` covers tt-metal program → op →
  op → tt-metal program in one session.
- *Broadcast*: none (unary).

## Design Conformance

- **Algorithm**: fp32-accurate `1/(1+exp(-x))` (Cody–Waite exp + LUT reciprocal + 2 Newton steps) with the
  `[-88, 88]` clamp, fused in DEST — as designed.
- **Pipeline / RISC ownership**: reader → `cb_input_tiles` → compute → `cb_output_tiles` → writer, implicit sync
  on both rings — as designed.
- **Work distribution**: flat page axis over the full grid (`device.compute_with_storage_grid_size()`), whole
  rounds of C pages, `split_work_to_cores(..., row_wise=True)`. Max pages per NEO = ⌈⌈n/C⌉/32⌉ = ⌈n/128⌉, the
  minimum, on every shape. Both dataflow halves issue every page without per-page barriers (implicit sync);
  rings are double-buffered. Fills the machine.
- **Thread plan**: read off the host program — `THREAD_PLAN = (2, 4, 2)` is the one constant; `num_threads` of
  reader / compute / writer = R / C / W; counts reach the kernels as CT args; loops are sized from them and
  `get_my_thread_id()`. Rings are sized per NEO (`BUFFER_DEPTH * block_tiles * C`). Both rings are implicit-sync
  (DM-filled/drained, one tagged transfer per entry), as the design's buffer table gives. Tests present and
  passing: every NEO ran (`probe_tag_neo`, odd NEOs compute sigmoid(-x), page→NEO mapping checked), plans
  (1,1,1)/(1,4,1)/(2,4,2)/(4,4,2) on 1/3/39/789 pages, one device session. The design states the bound
  (compute: ~57 SFPU instructions per row, matching toy_thread_plan's TOY_WORK=16 regime that scaled 2.7× with
  4 NEOs). This backend has no timing model, so the plan alternatives (1,4,1) / (1,1,1) were not measured;
  `pipeline_logs/model_cycles.jsonl` does not exist for this run.
- **Inter-core communication**: none (element-wise; no reuse-shared operand) — as designed.
- **Blocking-model fidelity**: knobs `THREAD_PLAN`, `BUFFER_DEPTH`, `DEST_TILES → block_tiles` each defined once
  in `sigmoid_program_artifacts.py`; ring sizes, loop bounds, CT args derive from them; no literal in a kernel
  (`8` in `SFPU_UNARY_CALL` is the SFPU's rows-per-face iteration count, not a block knob). No CB scales with a
  tensor dimension. Per-core block = the whole assignment (one chain call per NEO); the per-NEO compute loop is
  in DEST-capacity blocks, not per tile. Expression check: compute retires `block_tiles` per handshake; reader
  and writer use implicit sync (the DM0 handler posts credits per entry — the protocol the target page requires
  for NoC-filled rings, not a per-unit handshake the kernel issues). No finding.

## Registry Conformance

- `INPUT_TAGGERS = AXES` imported from the specification; no op-local axis function.
- `SUPPORTED` equals `PHASE0` axis for axis (all 7 axes, incl. `memory_config`, `sharding`, `alignment`, `rank`).
- `EXCLUSIONS = []`; `validate()` checks SUPPORTED per axis, then EXCLUSIONS; called first in the dispatch path
  (`sigmoid` → `_sigmoid` → `validate`), before any allocation or kernel work.
- The op file declares no `INVALID`.
- `python3 -m eval.validate_contract sigmoid --phase0`: PASSED, 0 errors (6 open items, all the spec's
  UNRESOLVED fields and the float32 entry's `ulp: TBD` second case — not op defects).
- XPASS auto-fixes: none needed.
- **INVALID audit** (spec): `INVALID = []`. Correct: one tensor, no structural impossibility in the rectangle;
  float32 × 16-bit DEST is legal-but-lossy and is a rule (R1 → EXCLUSIONS), not INVALID. The canonical
  bf8b + ROW_MAJOR entry is not needed: `bfloat8_b` is not in `TARGET["dtype"]` (and has no Quasar path).
  No weight axes. No issue to report.

## L1 Ledger Audit

- **Currency**: two rings in code (`cb_input_tiles`, `cb_output_tiles`), two rows in the ledger; size expression
  `BUFFER_DEPTH * block_tiles * C` × tensor page size matches `create_program_artifacts`. `block_tiles` bound
  and worst-case footprint updated for the full-sync fix.
- **Capacity vs live set**: each ring holds one sync block per NEO being worked + one in flight (double buffer);
  spans `neo` (×C, capacity is per NEO on STRIDED rings) and `sync_block` (×block_tiles), streams over `page`.
  No over- or under-sizing.
- **Page format vs DEST width**: both rings carry the tensor's own dtype (bf16): input is a dataflow payload that
  never passed through DEST; output is the ABI dtype packed from fp32 DEST. No wide page with 16-bit DEST, no
  narrow intermediate with fp32 DEST.
- **Disjoint lifetimes**: none — both rings live for the whole program, concurrently (pipelined); sharing
  answered in the ledger.
- **Bounds / closed form**: footprint `2 * BUFFER_DEPTH * block_tiles * C * page_size`, all symbols bounded;
  `num_tiles` appears in no capacity. Phase 0 default 131,072 B (3.3 % of 3,963,328 B); worst TARGET corner with
  full-sync DEST 524,288 B. Ring cap asserted on the host.
- **Data-movement budget**: each tensor crosses DRAM (or the L1 banks) once; 0 cross-core bytes; consistent with
  the implemented split; the cheapest split is the implemented one (R-shard deferred with a positive reason:
  `TARGET["sharding"]` is interleaved only).
- **Block-size defaults**: held — interleaved work spread across the full grid, then the coarsest block (whole
  assignment per core, DEST-capacity sync blocks).

## Precision Baseline

`tests/ttnn/unit_tests/operations/sigmoid/test_sigmoid_precision_baseline.py` (default config: bf16, TILE, DRAM,
HiFi4, fp32 DEST). Seed 42. ULP = distance on the bf16 grid to the float64 reference rounded to bf16. Ratio =
got / true over non-zero references (p05 / median / p95).

| Shape | Inputs | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | Max ULP | Ratio p05 / med / p95 |
|-------|--------|-----|-------------|--------------|------------------|---------|-----------------------|
| (1,1,32,64) | N(0,1) | 0.999991 | 1.95e-3 | 6.90e-4 | 4.19e-3 | 0 | 0.9974 / 1.0000 / 1.0026 |
| (1,1,32,64) | 10·N(0,1) | 0.999999 | 1.95e-3 | 3.34e-4 | 1.36e-3 | 0 | 0.9977 / 1.0000 / 1.0024 |
| (1,1,32,64) | U[0,1) | 0.999858 | 1.95e-3 | 9.74e-4 | 1.69e-2 | 0 | 0.9972 / 1.0000 / 1.0028 |
| (1,80,400) | N(0,1) | 0.999991 | 1.95e-3 | 6.97e-4 | 4.18e-3 | 0 | 0.9974 / 1.0000 / 1.0026 |
| (1,80,400) | 10·N(0,1) | 0.999999 | 1.95e-3 | 3.26e-4 | 1.33e-3 | 0 | 0.9977 / 1.0000 / 1.0024 |
| (1,80,400) | U[0,1) | 0.999860 | 1.95e-3 | 9.76e-4 | 1.68e-2 | 0 | 0.9972 / 1.0001 / 1.0029 |
| (2,4,128,512) | N(0,1) | 0.999991 | 1.95e-3 | 6.94e-4 | 4.17e-3 | 0 | 0.9974 / 1.0000 / 1.0026 |
| (2,4,128,512) | 10·N(0,1) | 0.999999 | 1.95e-3 | 3.28e-4 | 1.34e-3 | 0 | 0.9978 / 1.0000 / 1.0024 |
| (2,4,128,512) | U[0,1) | 0.999861 | 1.95e-3 | 9.72e-4 | 1.67e-2 | 0 | 0.9972 / 1.0000 / 1.0028 |
| (1,80,8400) | N(0,1) | 0.999991 | 1.95e-3 | 6.94e-4 | 4.17e-3 | 0 | 0.9974 / 1.0000 / 1.0026 |
| (1,80,8400) | 10·N(0,1) | 0.999999 | 1.95e-3 | 3.28e-4 | 1.34e-3 | 0 | 0.9978 / 1.0000 / 1.0024 |
| (1,80,8400) | U[0,1) | 0.999861 | 1.95e-3 | 9.73e-4 | 1.67e-2 | 0 | 0.9972 / 1.0000 / 1.0028 |
| (1,80,400), `dst_full_sync_en=True` | N(0,1) | 0.999991 | 1.95e-3 | 6.97e-4 | 4.18e-3 | 0 | 0.9974 / 1.0000 / 1.0026 |
| (1,80,400), LoFi + `math_approx_mode` | N(0,1) | 0.999991 | 1.95e-3 | 6.97e-4 | 4.18e-3 | 0 | 0.9974 / 1.0000 / 1.0026 |

**Assessment**: every element of every case is the correctly rounded bf16 value (0 ULP); all error is the bf16
output quantisation (max abs 1.95e-3 = half a bf16 ULP at [0.5, 1)). The ratio clusters at 1.0000 with a
symmetric ±0.3 % spread — rounding noise, no scale bug. The U[0,1) relative RMS (1.7e-2) is larger only because
the reference's std there is small (outputs in [0.5, 0.73]). Both saturated tails (10·N) are exact.

**Recommended tolerances**: defaults hold. bfloat16 cells are graded by `NUMERIC_RESOLVED` (≤ 1 / ≤ 2 ULP) and
the op meets it with margin (0 ULP). The shared `DEFAULT_TOLERANCES[(bfloat16, True)]` = (0.995, 0.04) is also
met by every case.

## Verifier CLI Summary

- supported_pass: 80 (74 PHASE0 cells + 6 NUMERIC regression cases)
- xfail_expected: 518 (74 bf16·ROW_MAJOR → Refinement 1; 296 bf16·16-bit-DEST and float32·fp32-DEST, both
  layouts → Refinement 2; 148 float32·16-bit DEST → permanent R1 refusal after Refinement 2)
- invalid_skipped: 0
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0
- no_axes_found: 1 (`test_op_loose[NOTSET]`: the empty `LOOSE_CASES` placeholder, skipped)
- Queue check (`--queue op_requirements.md`): 3 values in TARGET − SUPPORTED, 0 queue gaps.

## Recommendations

- **Queue**: TARGET − SUPPORTED has only three values (`float32`, `fp32_dest_acc_en=False`, `ROW_MAJOR_LAYOUT`).
  They bundle into two generality refinements (layout first: harder, and the dtype axis then extends both
  paths once), followed by one perf refinement on the 2:1 cadence. No further perf phase is filed: the work
  split is already at its structural minimum (⌈n/128⌉ pages per NEO), every DRAM crossing is the minimum, and
  the remaining perf lamps of the design (plan (1,4,1) vs (2,4,2), `block_tiles = 1`, `BUFFER_DEPTH = 3`,
  page-dealt split on tiny tensors) change no structural count — they need a timed backend (`ttsim-perf` /
  hardware) to show headroom or its absence.
- **float32 precision risk (Refinement 2)**: the float32 numeric entry is ≤ 2 float32 ULP. Phase 0's body has
  never been measured on float32 outputs; the reciprocal (LUT seed + 2 Newton steps) is the likely limiter.
  Measure early.
- **Perf headroom unconfirmed**: the design calls the op compute-bound; nobody has measured whether four NEOs
  already put it on the DM floor. If a timed backend shows it is DM-bound at (2,4,2), Refinement 3's SFPU cut is
  worth only what the 16-bit/1-NEO plans gain from it.
- **Hardware constants in Python**: `FREE_DM_THREADS = 6`, `NEOS = 4`, `MAX_RING_BYTES` are target-page numbers
  with no Python query; they are used only in host-side legality checks, not in sizing. Acceptable; replace with
  queries if tt-metal exposes them.
- **Sharding** is outside TARGET; R-shard stays a knob-turn for when it is added.
