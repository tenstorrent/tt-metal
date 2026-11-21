# Verification Report: typecast

Target: Quasar (`quasar`, 8×4 clusters) on the ttsim simulator. Specification: `eval/specs/typecast/spec.py` (sha256 56aabff163b0…). Specification lint (`python3 -m eval.validate_contract typecast --phase0`): PASSED, 0 errors (5 UNRESOLVED open items, all the specification's and none binding: `tt_regen_spec.governs`, `NUMERIC.special_values`, `NUMERIC.freedoms`, `PERFORMANCE.bar`, `PERFORMANCE.measure`).

## Code Review

What was reviewed: `typecast.py`, `typecast_program_artifacts.py`, the three kernels, `op_design.md`, `l1_ledger.md` and the tests. The implementation is clean and close to the design. Fixes made:

1. **R7 test gap (fixed).** R7's check says "a sharded input is a test the op carries itself", and the op carried none. Added `test_sharded_input_refused_by_name`: a height-sharded L1 input raises `NotImplementedError` now (Phase 0 refuses it on `memory_config` first). With `"L1"` temporarily admitted to SUPPORTED, the refusal must name `sharding='height'`, which pins R7 for Refinement 1, the point where L1 becomes supported.
2. **Stale design statement (fixed).** `op_design.md` said the DEST section puts "none" on `block_tiles`, and its worked numbers used `block_tiles = 8`. The eltwise chain clamps a `PerBlockSize` block to DEST and syncs the rings per clamped window, which straddles the ring end. The code caps the block at `dest_tiles()` = 4 and `static_assert`s it in the compute kernel. The ledger already said this, so I corrected the Mechanism-caps row, the `block_tiles_cap` formula and the three worked values in the design. The next stage would otherwise plan against a cap that does not exist.
3. Removed a stray `# noqa: E402` on the specification import in `typecast.py` (cosmetic).

Nothing deferred to a refinement from code review.

### Specification rules

| Rule | Applies at Phase 0? | Status |
|---|---|---|
| R1 float→int saturate | no (no integer dtype supported) | re-arms in Refinement 1; NUMERIC regression cases named in its Done-when |
| R2 int→int wrap | no | re-arms in Refinement 1 (flagged: SFPU arms may saturate or be missing) |
| R3 one step | yes (trivially: bf16↔fp32 goes unpack → 32-bit DEST → pack, no third format) | met; int32→uint32 cells check it in Refinement 1 |
| R4 DRAM or L1 | partially (DRAM only in PHASE0) | the code addresses both through the TensorAccessor; L1 in Refinement 1 |
| R5 refuse outside SUPPORTED with `NotImplementedError` | yes | met: 2285 xfail_expected, 0 xfail_wrong_mode |
| R6 output in the input's buffer, interleaved | yes | met: `TensorSpec(..., buffer_type=input.memory_config().buffer_type)`; `test_output_in_input_buffer` checks DRAM interleaved; the L1 check is in Refinement 1's Done-when |
| R7 axes from `memory_axes`, sharded refused by name | yes | met; test added (fix 1) |
| X1 non-DataType `dtype` | yes | `TypeError` before any axis is derived; golden `test_rejection[X1]` passes |

None of the rules has a `structure` part. GUIDANCE is followed: the same-dtype call goes through the one program and returns a fresh tensor, an L1 input is read through the accessor, and padding is handled as pages.

## Registry Conformance

- `INPUT_TAGGERS = AXES`, imported from `eval.specs.typecast.spec`. No op-local axis function.
- `SUPPORTED` equals `PHASE0` on all seven axes (`dtype`, `target_dtype`, `layout`, `memory_config`, `sharding`, `alignment`, `rank`); `memory_config` and `sharding` come from `eval.memory.memory_axes` (R7).
- `EXCLUSIONS = []`. `validate()` checks X1 first, then SUPPORTED per axis (`UnsupportedAxisValue`), then EXCLUSIONS (`ExcludedCell`). The entry point calls it as its first line.
- The op file declares no `INVALID`.
- No XPASS drift, so no auto-fix to SUPPORTED was needed.
- **INVALID audit (specification)**: one entry, `{dtype: bfloat8_b, layout: ROW_MAJOR}`, the canonical block-float + row-major activation entry. It is a single-tensor coupling and a structural impossibility. No cross-tensor entries; the op is not norm-like, so no weight canonicalization applies. No issues.

## Design Conformance

- **Algorithm / pipeline**: as designed. Regime `tile_stream_float`: reader → `cb_input_tiles` → `compute_kernel_lib::typecast` (32-bit DEST; UnpackToDest for fp32, UnpackToSrc for bf16; `DataFormatReconfig::Disabled`) → `cb_output_tiles` → writer. One `ttnn.generic_op` per call, Metal 2.0 `ProgramSpec` only.
- **Work distribution**: `split_work_to_cores(grid, tensor_tiles, row_wise=True)` with contiguous ranges. All 32 clusters are engaged once `tensor_tiles ≥ 32`; the Phase-0 calls (8 and 16 tiles) engage 8 and 16 clusters at one tile each. The machine is filled as designed. Both dataflow halves use the same per-entry implicit-sync transport, the protocol the target page prescribes.
- **Thread plan**: read off the host program as `THREAD_PLAN = (1, 1, 1)`, one constant, passed as CT args `reader_threads` / `compute_threads` / `writer_threads`. Reader and writer stride by R / W from `get_my_thread_id()`; compute takes `num_tiles / C + (t < num_tiles % C)`. Both rings hold `BUFFER_DEPTH · block_tiles · C` entries, so they are sized per NEO. The design states the op as data-movement bound, with the reason that bf16↔fp32 is an SFPU no-op (`typecast.h:19-23`): copy and pack only. Required tests are present and pass: the plan read off the ProgramSpec (`test_thread_plan_on_the_program`); the device session of another program, then the op, then `ttnn.to_layout` (`test_device_session_order`); and C > 1 plans (1,2,1), (1,4,1), (2,2,2), (4,4,2) bit-exact on 93 tiles. The compute loops are sized from C, so a missing thread would leave tiles unconverted. Not measured against alternatives: ttsim has no timing model, and no `model_cycles.jsonl` exists for this phase.
- **Blocking-model fidelity**: `block_tiles`, `BUFFER_DEPTH`, `THREAD_PLAN` and `grid` are each defined once. `block_tiles` is a compute CT arg, and both rings, the cap and the helper's `.block_size()` derive from it. No CB scales with a tensor dimension. The DEST cap is a mechanism cap (the chain's sync window), not a collapsed knob. The per-core loop is not size-1: the block is the coarsest that keeps two blocks in flight, up to 4. Expression: compute syncs once per block. Reader and writer issue one tagged call per ring entry, which is the implicit-sync transport's unit (target page) and not a per-unit handshake: there is no reserve, barrier or push per tile.
- **Inter-core communication**: none, as designed (no dependent or reuse-shared axis).

## L1 Ledger Audit

- **Currency**: both declared rings (`cb_input_tiles`, `cb_output_tiles`) have rows. Size expressions (`BUFFER_DEPTH · block_tiles · C` entries of `in_page` / `out_page`) match `create_program_artifacts`. The ledger already carries the implementer's DEST-cap correction.
- **Capacity vs live set**: both rings are double-buffered by a stated overlap decision (`buffer_depth = 2`), with the `tile` axis spanned at `block_tiles` and the folded axes streamed. No over-capacity, no collapsed extent.
- **Page format vs DEST width**: 32-bit DEST is on. The Float32 input page is required (UnpackToDest of all 32 bits). The Float16_b output page is the output ABI (the narrowing is the conversion itself), not a truncated accumulation. No finding.
- **Disjoint lifetimes**: none; both rings are live for the whole pipeline, and the justification cell explains why they cannot share (different formats and producers).
- **Bounds / closed form**: total `C · buffer_depth · block_tiles · (in_page + out_page)`; every symbol is bounded in the symbol table. Worked values on quasar: Phase-0 calls 12–16 KiB; at the cap (C=4, fp32→fp32) 256 KiB, against 3.9 MB available.
- **Data-movement budget**: input 1×, output 1×, 0 cross-core, consistent with the implemented split S1. The cheapest-traffic split is implemented, since every split of an elementwise op moves the minimum.
- **Block-size defaults**: interleaved default held: the split spreads over the full grid, then takes the coarsest block that keeps `buffer_depth` blocks in flight, clamped by the measured DEST mechanism cap.

## Precision Baseline

`tests/ttnn/unit_tests/operations/typecast/test_typecast_precision_baseline.py`: 4 pairs × 4 shapes, standard-normal inputs (the spec's distribution), seed 42. PCC and abs errors are measured against the float64 input (the specification's reference). Relative RMS = RMS error / std(reference). ULP is in the output dtype. "Bit mism." counts elements whose bits differ from torch's `Tensor.to`.

| Pair | Shape | Bit mism. | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | Max ULP | got/true ratio p5 / median / p95 |
|---|---|---|---|---|---|---|---|---|
| bf16→bf16 | all 4 shapes | 0 | 1.0 | 0 | 0 | 0 | 0 | 1 / 1 / 1 |
| bf16→fp32 | all 4 shapes | 0 | 1.0 | 0 | 0 | 0 | 0 | 1 / 1 / 1 |
| fp32→fp32 | all 4 shapes | 0 | 1.0 | 0 | 0 | 0 | 0 | 1 / 1 / 1 |
| fp32→bf16 | (1,4,32,64) | 0 | 0.9999986 | 7.77e-3 | 1.12e-3 | 1.65e-3 | 0.5 | 0.99738 / 0.99999 / 1.00259 |
| fp32→bf16 | (1,1,128,128) | 0 | 0.9999986 | 7.80e-3 | 1.13e-3 | 1.66e-3 | 0.5 | 0.99739 / 1.00001 / 1.00261 |
| fp32→bf16 | (4,8,32,256) | 0 | 0.9999986 | 1.54e-2 | 1.13e-3 | 1.66e-3 | 0.5 | 0.99737 / 0.99999 / 1.00262 |
| fp32→bf16 | (1,1,2048,256) | 0 | 0.9999986 | 1.54e-2 | 1.13e-3 | 1.66e-3 | 0.5 | 0.99737 / 1.00000 / 1.00262 |

**Assessment**: every supported cell is bit-exact against torch, which is what the specification's `exact` metric requires. The fp32→bf16 errors are torch's own round-to-nearest-even rounding: ≤ 0.5 bf16 ULP, with the ratio spread centred on 1.0 (no scale bug). The mechanism tests also pin RNE on crafted ties (low 16 bits 0x8000 with both LSB parities, plus their neighbours) and show that fp32→fp32 keeps the low 13 mantissa bits (UnpackToDest, not SrcA).

**Recommended tolerances**: exact (every supported cell resolves to `exact` in `NUMERIC_RESOLVED`).

## Verifier CLI Summary

Golden suite: `eval/eval_test_runner.sh eval/golden_tests/typecast/` (8 sim workers, forked, 3m19s). Report: `verifier_report.json` (copy in this directory).

- supported_pass: 24
- xfail_expected: 2285
- invalid_skipped: 832 (224 INVALID bf8b+RM cells, plus bfloat8_b cells skipped as `unbuildable on quasar`)
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0
- no_axes_found: 2 (`test_rejection[X1]` passed; `test_op_loose[NOTSET]` skipped because LOOSE_CASES is empty)
- queue_gaps (`--queue op_requirements.md`): 0

The 5 NUMERIC regression cases (float32 → uint8 / uint16 / int32) are expected refusals until Refinement 1.

## Queue Summary

TARGET − SUPPORTED, and where each value goes:

| Axis | Missing | Disposition |
|---|---|---|
| dtype / target_dtype | int32, uint32, uint16, uint8 | Refinement 1 |
| dtype / target_dtype | bfloat8_b | Omitted: no block-float path on quasar (target page); its cells are unbuildable here |
| memory_config | L1 | Refinement 1 (tile-path knob-turn) |
| rank | 2, 3 | Refinement 1 (tile-path knob-turn) |
| alignment | w_non_aligned, h_non_aligned | Refinement 1 (tile-path knob-turn) |
| layout | ROW_MAJOR | Refinement 2 |

Refinement 3 is the perf slot on the 2:1 cadence, gated on a structural count because the backend has no clock. No LOOSE_CASES entry is perf-flagged, so I chose the region: the SFPU integer pairs at `[1,1,2048,256]`, whose full contract Refinement 1 lands. Every refinement is a knob-turn or an extension of the one built regime. There is no scheme-change candidate: no dependent axis, no shared operand, no streaming rewrite.

Only two generality refinements are filed, fewer than the usual 3–5, because that is the whole TARGET gap. L1 / rank / alignment on the tile path need no kernel change and are too small to stand alone, so they are bundled into Refinement 1.

## Recommendations

- **No further perf refinement filed after Refinement 3 (headroom gate).** The design's other perf lamps cannot show a structural gain on ttsim, and each needs a timed backend (`ttsim-perf`):
  - shrinking the grid for small tensors: the Phase-0 calls run 8 and 16 clusters at one tile each;
  - the same-dtype DM-only program (regime `same_dtype_dm_copy`): the compute pass is off the critical path of a data-movement-bound op;
  - the block size.

  The op already moves the minimum bytes over the full grid. Revisit these if a `ttsim-perf` run is made.
- **Refinement 1 is the risky one.** Its Verifier notes list four open questions, and the first two decide the design of the integer path, so settle them before writing kernels:
  - uint16 / uint32 cannot be ring formats;
  - per-source unpack modes;
  - SFPU arms that are missing, or that saturate where R2 requires a wrap;
  - the RNE grading of int→bf16.
- **Refinement 2 has no wrapper fallback**: the run's one-dispatch rule rules out `ttnn.to_layout`. Non-aligned narrow sticks (17 B for uint8 W=17) are the hard case.
- L1 pressure: none. The worst case at the DEST cap with C=4 is 256 KiB of 3.9 MB available, so no OOM is foreseeable for any TARGET shape.
