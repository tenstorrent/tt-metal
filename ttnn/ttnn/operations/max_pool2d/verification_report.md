# Verification Report: max_pool2d

Target: Quasar (8×4 clusters, 4 NEOs + 6 user DM cores each), ttsim simulator. Host program: Metal 2.0 `ttnn.ProgramSpec`, one `ttnn.generic_op` per call. Specification: `eval/specs/max_pool2d/spec.py` (lint: 0 errors, 5 UNRESOLVED open items, none blocking).

## Code Review

Fixed:

1. **Rule R9: shard orientation of the result (MUST violation, fixed).** `plan_output` copied the input's shard orientation onto the result. R9 says *"The shard orientation is row-major"*. A probe with a col-major input on a 2×4 grid (height- and width-sharded) returned exact values in a `COL_MAJOR` result. The runner builds only row-major shards, so no golden cell could see this. Fix:
   - the result is now always `ROW_MAJOR` on the input's grid, its shards dealt in row-major core order;
   - `Shard.in_core` records the core holding input shard g (in the input's own order);
   - an owner is IN_PLACE (zero-copy from its own width shard) only when it also holds input shard g; otherwise it stages that channel run over the NoC.
   
   New test: `test_max_pool2d_plan.py::test_col_major_input_gives_row_major_result` (height and width × TILE and ROW_MAJOR), all exact.
2. Dead code: the unused host constant `FACE` was removed.
3. `l1_ledger.md`: two statements disagreed with the code and were corrected (below).

Reviewed and found conformant (no change needed):

- **Registry**: `INPUT_TAGGERS = AXES` (imported from the specification), `SUPPORTED` == `PHASE0` axis for axis, `EXCLUSIONS = []`, and `validate()` checks the R2 refusal, then SUPPORTED per axis, then EXCLUSIONS, raising `UnsupportedAxisValue` / `ExcludedCell`. The entry point calls `validate()` first. No `INVALID` in the op file. Malformed calls (X1–X4) raise `ValueError` before the axis gate.
- **Rules**:
  - R1: padded taps re-read a real tap of the same window, so padding is never stored; the `all_negative` regression passes.
  - R2: refused natively.
  - R3: `default_compute_kernel_config()` is the only default; lint `default_factory` passes.
  - R4: `validate()` reads neither `math_fidelity` nor `math_approx_mode`; the fidelity is forwarded to `ComputeHardwareConfig`.
  - R5: lists and tuples are canonicalized; the `yolo_sppf_lists` loose case passes.
  - R6: rows are read and written as the specification maps them.
  - R7: in TILE, rows past N·H·W and columns past C are never addressed.
  - R8: `memory.memory_axes()`.
  - R9: placement checked in every cell by `HOOKS["check"]`; orientation fixed above.
  - R10: one-core height and width tests, and pool-of-a-pool tests.
  - R11: always ROW_MAJOR, the input's dtype.
  - R12: holds.
- **Design conformance**: the algorithm (stream tiles + staged receptive field + DEST max fold), the pipeline topology (reader ×R stage + gather → compute ×C fold → writer ×W store) and the work distribution (R9 output shards × whole-round position sub-ranges on spare cores) all match `op_design.md`. Measured coverage: SPPF fills all 32 clusters; ResNet height-sharded runs on the 32 input-shard cores.
- **Thread plan**, read off the host program:
  - `THREAD_PLAN = (4, 4, 2)` is one host constant, passed as CT args `reader_threads` / `compute_threads` / `writer_threads`, and `KernelSpec(num_threads=...)` takes R, C, W from it.
  - Rings are sized per compute thread (`window_taps` = `2·taps_per_batch·C`, `pooled_tiles` = `2·C`).
  - Both rings are explicit-sync and listed in `disable_dfb_implicit_sync_for`; every transfer is tagged trid 0, and each DM kernel hands the tag back.
  - The proofs are present and passing: every-thread-ran (`test_every_neo_owns_a_real_tile`: negative input over memory left holding a positive result); one device session after another program and before tt-metal's own `to_layout` / `to_memory_config` (`test_one_device_session_with_tt_metal_programs`); other plans (`test_thread_plans`).
  - The design's bound is stated as compute (K = 9–25 unpack + SFPU passes per output tile), so C = 4 is the design's own choice.
  - ttsim has no timing model, so no plan-vs-alternative measurement applies; there is no `model_cycles.jsonl` for this phase.
- **Blocking-model fidelity**:
  - Every knob is one named host constant or a value derived from one: `TAP_RING_DEPTH`, `OUT_RING_DEPTH`, `STAGING_DEPTH`, `MIN_ROUNDS_PER_CORE`, `MAX_POSITION_SPLITS`, `L1_RESERVE_BYTES`, `taps_per_batch` (CT), `block_tiles` (RT per assignment).
  - No CB scales with a whole-op dimension. The staging scratchpad is the design's budget-predicated resident block, with the staged-streamed fallback built and pinned by `test_staged_streamed`.
  - The receptive-field formula appears in both host (`Geometry.receptive_sticks`) and kernel (`receptive_field`). It is geometry, not a knob, and is commented as a pair; left as is.
  - Expression check: the reader stages once per block and gathers one batch (`taps_per_batch` = K) per reserve/barrier/push. The compute waits and pops per tap inside one held DEST fold (a phase-internal loop; indexed `copy_tile` on a STRIDED ring is unverified on Quasar). The writer handshakes and barriers per output tile, as the design's block table states. Both per-unit handshakes are the design's perf lamps and are queued as Refinement 5.
- **Helper usage**: the design's rejection of `eltwise_chain` / `binary_sfpu<BinaryMax>` (not available on Quasar) and of `reduce<MAX>` (within-tile reduce) holds; the raw `copy_tile` + `binary_max_tile` fold is justified in the kernel header. No multicast, so no `McastArgs` question. TensorAccessor, `kernel_main()` and `api/...` include paths are all correct.
- **Correctness**: push equals pop per NEO (`window_taps`: Q·K pushed in `taps_per_batch` units, Q·K popped in 1s; `pooled_tiles`: Q per NEO). Padded rounds keep the per-thread entry counts equal.

Advisories (GUIDANCE, not blocking):

- GUIDANCE asks for one accessor data path for every placement. The op adds a second path: an in-place `LocalTensorAccessor` path for a width-sharded owner core. That departure is the zero-copy consumption of a resident local shard and is correct.
- For height-sharded inputs, a core's own rows are staged through the accessor (a core-local copy) together with the halo. The design defers in-place own rows (`in-place-own-rows` regime row) with a positive reason: it saves 1 of K+2 core-local passes, while the staged copy moves each halo row across cores once instead of up to kh times. Height-sharded pooling needs neighbour rows, so it is not plain local sharding.

Deferred to refinements: none of the review findings (all fixed in place).

## Registry Conformance

- Confirmed: `INPUT_TAGGERS`, `SUPPORTED`, `EXCLUSIONS` and `validate()` are present and correctly wired in `max_pool2d.py`, and the op file declares no `INVALID`.
- No SUPPORTED auto-fix was needed: `xpass_drift = 0`. `validate_contract --phase0` passes, so SUPPORTED == PHASE0.
- INVALID audit (specification): two entries, `{DRAM, height}` and `{DRAM, width}`. Both couple the two placement axes of the same tensor (`input_tensor`) and are structural (the runner builds no DRAM-sharded tensor; eval/CONVENTIONS.md §2.3). There are no cross-tensor couplings and no "not yet" entries. The canonical bf8b + ROW_MAJOR entry is not needed: `bfloat8_b` is not in TARGET (removed by the specification's DECISIONS, and Quasar has no block-float path). There are no weights, so no canonicalization cells.

## L1 Ledger Audit

- **Ledger currency**: the three rows (`window_taps`, `pooled_tiles`, `staging` scratchpad) match the code's declarations and size expressions. Two corrections were made in place:
  - the staging size when every core is IN_PLACE is 16 B (one alignment unit, so the binding exists), not 0 B;
  - `window_taps`' page-width reason is a dataflow-produced payload (NoC copies; no pack from DEST to truncate). The earlier wording said it never passes through DEST, which is wrong: it is unpacked into DEST.
- **Capacity vs live set**:
  - Over: both rings exceed their live set by exactly their stated depth knob (2, double buffering).
  - Under: `window_taps` spans the taps axis with capacity `taps_per_batch` (scales); staging spans the block's receptive field and scales with `block_tiles`.
  - Nothing flagged.
- **Page format vs DEST width**: there are no Float32 pages with fp32 DEST off. `pooled_tiles` is bf16 under fp32 DEST, which is the *under* case, answered by the mechanism: the max selects an input element exactly and the output dtype is fixed by R11, so nothing accumulates and nothing is truncated. `window_taps` is input bytes.
- **Disjoint lifetimes**: none. All three buffers are concurrently live, and the sharing column justifies each one.
- **Bounds / closed form**: `taps_per_batch` is bounded by the ring cap; `block_tiles` by the budget predicate (device query minus `L1_RESERVE_BYTES`). The total `C·tile_bytes·(2·taps_per_batch + 2) + staging_bytes(block_tiles)` is closed-form.
- **Per-core footprint**: worked maximum ≈ 440 KB (SPPF, K = 25) against a 3.9 MB budget. The ring terms scale with `taps_per_batch` and C; the staging term scales with `block_tiles`, `stick_pitch` and W.
- **Block-size defaults**: interleaved spreads the units over the grid with one block per core (`block_tiles = Q`, the coarsest). Sharded is one block per core. There is no departure, and the staged-streamed fallback is reached only under a test override.
- **Data-movement budget**: present and consistent with the implemented split. The cheapest-traffic split (channels only, `MAX_POSITION_SPLITS = 1`) is implemented as a knob, but the default spends 0.65 × |in| more DRAM reads on SPPF to put 4× the NEOs to work, and that justification is occupancy plus "compute-bound at K = 25". Disposition: folded into **Refinement 3**. That refinement changes the per-tile compute cost and must re-record the row with both counts and a positive reason (noted in the ledger's verifier-audit section).

## Precision Baseline

`tests/ttnn/unit_tests/operations/max_pool2d/test_max_pool2d_precision_baseline.py`, bfloat16, inputs drawn from N(0, 1) with seed 42, default config (HiFi4, fp32 DEST):

| Shape | Layout / placement | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | ULP (max) | got/true median [p5, p95] |
|-------|-------------------|-----|-------------|--------------|------------------|-----------|---------------------------|
| 1×8×8×64, 5×5/s1/p2 | ROW_MAJOR, L1 interleaved | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 1.0 [1.0, 1.0] |
| 1×20×20×256, 5×5/s1/p2 (SPPF) | ROW_MAJOR, DRAM interleaved | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 1.0 [1.0, 1.0] |
| 1×32×32×64, 3×3/s2/p1 (ResNet) | TILE, L1 height-sharded (2 cores) | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 1.0 [1.0, 1.0] |
| 1×32×64×32, (3,5)/(2,1)/(1,2) | TILE, DRAM interleaved | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 1.0 [1.0, 1.0] |

**Assessment**: bit-exact on every shape. Every output element is the input element torch selects, as the movement class requires. There is no scale signature (the ratio is exactly 1.0 everywhere).
**Recommended tolerances**: exact (every cell resolves to `exact` in `NUMERIC_RESOLVED`).

## Verifier CLI Summary

Golden suite (`eval/eval_test_runner.sh`, 8 sim workers, forked), after the R9 fix: 72 passed, 0 failed, 480 skipped, 896 xfailed, 0 hangs.

- supported_pass: 68 (64 PHASE0 cells + 1 loose case + 3 numeric regressions)
- xfail_expected: 896
- invalid_skipped: 480
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0
- no_axes_found: 4 (the X1–X4 rejection tests, which the registry does not drive)
- Queue coverage (`--queue op_requirements.md`): 7 values in TARGET − SUPPORTED, 0 queue gaps.

`verifier_report.json` is copied next to this report.

Unit tests: `tests/ttnn/unit_tests/operations/max_pool2d/`: 49 passed (25 acceptance, 20 plan / regime tests including the new orientation test, 4 precision baseline).

### Scratch experiment (not committed): SUPPORTED opened to TARGET

To size the queue, the verifier temporarily set `SUPPORTED = TARGET` and ran the golden suite (results only; the file was restored): 446 passed, 282 failed, 0 hangs.

- **bfloat16**: 439 / 481 cells pass, covering `batched`, `disjoint`, `pad=none`, `dilated`, `fp32_dest_acc_en=False`, and non-tile-aligned C in ROW_MAJOR and in TILE width-sharded. The 42 failures are all TILE + non-tile-aligned C in interleaved or height-sharded placement, refused by the op's own `c_core % 32` guard. Goes to Refinement 1.
- **float32**: 0 / 240 pass. 219 fail at program build: an FP32 DFB consumed with 32-bit DEST needs an explicit `unpack_modes` entry. 21 hit the same TILE non-aligned guard. Goes to Refinement 2.

## Recommendations

- **Queue order**: Refinement 1 (geometry + non-aligned C, `/memory-layouts`), then Refinement 2 (float32 + 16-bit DEST, `/numeric-formats-metal`). Those two cover all 7 missing TARGET values. Then the perf phases: Refinement 3 (SPPF separable window, T3), Refinement 4 (ResNet: threads sized to the work and TILE gather transactions), Refinement 5 (block / depth co-tune). Generality runs out after Refinement 2, so the remaining phases are all perf.
- **Perf gates on ttsim** are structural counts, with before values in each entry:
  - SPPF: 25 tap tiles folded per output tile;
  - ResNet 32-shard: 1,152 gather reads per core and 3 padded recomputes per core;
  - SPPF: 1 DEST section, 1 writer barrier and 25 compute ring waits per output tile.
  
  None of these is a time result. A timed backend must confirm each one.
- **Occupancy observation (no failing cell)**: for the ResNet call height-sharded over 32 cores, each core has half a stream tile of work. Under the (4, 4, 2) plan, 3 NEOs and 3 readers duplicate that tile (queued in Refinement 4).
- **L1**: footprint is far below budget on every TARGET input (max ≈ 440 KB of 3.9 MB). Nothing reaches the staged-streamed regime except the test override.
- **Specification advisory (not a defect)**: R10's one-core requirement and R9's row-major orientation are not reachable by the generic runner (it builds row-major shards sized for the largest divisor of the core count). The op's own tests now cover both.
