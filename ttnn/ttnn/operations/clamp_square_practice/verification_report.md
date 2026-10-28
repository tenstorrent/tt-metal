# Verification Report: clamp_square_practice

Board: Blackhole (12×10 = 120-core compute grid). Date: 2026-10-28.

## Blocker to note first: no golden suite

`eval/golden_tests/clamp_square_practice/` does **not exist** (no `feature_spec.py`, no `test_golden.py`). The
planner noted the same at design time (`op_design.md` → Overview). Consequences:

- `eval/eval_test_runner.sh` + `python3 -m eval.verify_supported` **could not be run**: there is no test directory
  to run, and no `test_axes.json` / `INVALID` source to join against. **No `verifier_report.json` was produced.**
  The registry contract was instead checked by hand (below) and exercised by the acceptance test's rejection cases.
- `TARGET` and `INVALID` are unavailable. The refinement queue in `op_requirements.md` uses the **provisional TARGET**
  from `op_design.md` → "TARGET axes and the regime table" (the planner's list of unlockable axis values), stated
  explicitly in that file.
- **Action for the user:** author the suite via `/golden-tests` (or by hand) before running the refinement loop.
  Recommended `INVALID` content: `{dtype: bfloat8_b, layout: ROW_MAJOR_LAYOUT}` (block-float has no RM form). This op
  has no weight tensors, so no cross-tensor or no-weight canonicalization entries apply. Use the axis names
  `dtype / layout / memory_layout / buffer_type / rank / alignment` that the op file already uses, so nothing needs
  remapping.

## Code Review

Fixed:

1. **Out-of-fp32-range bounds escaped as `OverflowError`.** `validate()` accepted any finite Python float. The host
   then packed the bounds with `struct.pack("<f", v)`, which raises `OverflowError` (neither `ValueError` nor
   `RuntimeError`) for `|v| > FLT_MAX`, e.g. `min_val=-1e300`. `torch.clamp` refuses such scalars as well
   ("value cannot be converted to type float without overflow"), and the design specifies "finite fp32". So
   `_check_bound` now raises `ValueError` for `|v| > FLT_MAX`. Regression tests are in
   `test_clamp_square_practice_extended.py`: rejection at ±1e300 and 1e39, `±FLT_MAX` accepted, and
   NaN/inf/str/bool rejected.

Reviewed with no defect found:

- **Prompt rules.** `eval/prompts/clamp_square_practice.txt` has no `## Rules` section. Its mandated behavior holds:
  the import path and signature are exact, all six rejection classes raise. `UnsupportedAxisValue` derives from
  `NotImplementedError ⊂ RuntimeError`, and `min>max` raises `ValueError`. Phase 0 is multi-core. Each call is one
  `ttnn.generic_op` dispatch, and output allocation is not a dispatch.
- **Helper usage.** Compute is a single `eltwise_chain`: `CopyTile → Clamp → Square → PackTile`, `FullBlock`, with
  `PerBlockSize` wait/pop/reserve/push. Element inits and reconfig run once per core, and Clamp → Square is fused in
  DEST with no intermediate CB. `unary<>`, FPU `square<>` and the packer-side activations were all correctly
  rejected in the design. No raw compute API remains apart from the required `compute_kernel_hw_startup`.
- **Correctness.** Each CB's push count equals its wait count: every side uses the nominal `block_tiles` quantum,
  including the ragged tail. Dataflow uses `TensorAccessor`, `void kernel_main()`, and the
  `api/dataflow/dataflow_api.h` include. Bounds are passed as fp32 bit patterns, not truncated to bf16 by `<<16`,
  and they are RT args, so a new bound pair does not recompile.
- **Broadcast.** None. The scalars are RT args, so no constant CB is filled.

## Registry Conformance

- `INPUT_TAGGERS` (`rank`, `alignment`) use the `(inputs, axes)` signature.
- `SUPPORTED` covers every gated axis: `dtype`, `layout`, `memory_layout`, `buffer_type`, `rank`, `alignment`.
- `EXCLUSIONS = []`.
- `validate()` checks SUPPORTED per axis, then EXCLUSIONS, then the scalar checks. It is the first line of the public
  entry point.
- The op file does **not** declare `INVALID`.
- No XPASS evidence was available, so SUPPORTED was not auto-fixed.
- The INVALID audit could not be done because `feature_spec.py` is absent (see the blocker above).

## Design Conformance

| Dimension | Design | Implementation | Verdict |
|---|---|---|---|
| Algorithm | Fused Clamp → Square in DEST | Same, one chain | ✓ |
| Pipeline | reader (NCRISC/NoC0) → compute → writer (BRISC/NoC1) | Same | ✓ |
| Work split | Flattened tile axis, `split_work_to_cores(row_wise=True)`, full grid | Same. 120/120 cores at ≥120 tiles (measured) | ✓ fills the grid |
| Dataflow batching | `block_tiles` reads → 1 barrier. Writes → 1 flush per block, 1 final barrier | Same on **both** halves | ✓ |
| Inter-core | None | None | ✓ |

**Blocking-model fidelity.**

- `BLOCK_TILES_DEFAULT`, `INPUT_DEPTH_BLOCKS`, `OUTPUT_DEPTH_BLOCKS` and `core_grid` are each defined once in the
  program descriptor and are builder parameters.
- `block_tiles` is derived once, with `min(default, dest_limit(FP32_DEST_ACC_EN, DST_FULL_SYNC_EN), core_tiles_g1)`,
  and passed as **one** CT arg to all three kernels.
- CB capacities are `depth · block_tiles · page`, so no CB depends on an op dimension.
- `dest_limit()` and the compute config read the same two flags. The kernel `static_assert(block_tiles <=
  DEST_AUTO_LIMIT)` backstops the host mirror.
- **Expression.** The planned extent is >1. Reader and writer acquire and retire per block, not per tile. The tile
  loop inside `load_block` / `store_block` only issues async commands under one barrier or flush. Compute syncs per
  block. No expression failure.
- **Half-turned split.** None: the per-core loop runs at the block, and the block is pinned at the DEST cap.
- One design-level coupling is noted as a *perf lever*, not a defect: the NoC batch size and the DEST group are the
  same knob, `block_tiles ≤ 8`. Decoupling them (multi-block issue before one barrier) is the main lever of the
  Refinement 3 perf entry.

## L1 Ledger Audit

- **Currency.** Two CBs are declared (`cb_input_tiles` = 0 and `cb_output_tiles` = 1), and `l1_ledger.md` has exactly
  these two rows. The size expressions `depth · block_tiles · 2048` match the descriptor.
- **Capacity vs live set.** Capacity = 2 × live set on both CBs, for double buffering, as stated. Neither CB is
  oversized. Both rows **span** the `tile` axis, and their capacity scales with `block_tiles`, so neither extent is
  collapsed.
- **Page format vs DEST.** Float16_b pages with `fp32_dest_acc_en=False`. Not over-wide and not truncating.
- **Disjoint lifetime.** The two CBs have concurrent lifetimes (pipelining) and their non-sharing is justified.
- **Bounds / closed form.** The symbol table bounds every symbol, and the total `L1 = (in_depth + out_depth) ·
  block_tiles · 2048` is 64 KiB at Phase 0. Both terms scale with `block_tiles × depth`. No op dimension appears.
- **Data-movement budget.** The ledger budget of `2T` DRAM and 0 cross-core matches the implemented split. That is
  the cheapest split, and it is the one implemented. The `sharded_resident` row is deferred for a positive reason:
  the Phase-0 contract excludes sharded inputs. It becomes Refinement 1.
- **Block-size defaults.** Interleaved means the full grid plus the coarsest block that fits. The coarsest block is
  capped by the DEST mechanism (8), not by L1. This holds.
- No ledger findings needed fixing or filing.

## Precision Baseline

`test_clamp_square_practice_precision_baseline.py`. Input is `randn·2` in bf16. The reference is the fp32
`torch.clamp(x, lo, hi)**2`. ULP is measured against the bf16-rounded reference. The table below shows bounds
(-1, 1) and (-0.3, 0.7), which are identical across all four shapes; the third bound set, (-4, 4), is at the end of
this section.

| Shape | Bounds | PCC | Max Abs Err | Mean Abs Err | Rel RMS Err | Max ULP | Exact frac | got/true median [p5, p95] |
|---|---|---|---|---|---|---|---|---|
| (1,1,32,32) | (-1, 1) | 0.9999994 | 1.94e-3 | 1.61e-4 | 4.93e-4 | 0 | 1.000 | 1.0000 [0.9983, 1.0023] |
| (1,1,256,256) | (-1, 1) | 0.9999994 | 1.94e-3 | 1.56e-4 | 4.85e-4 | 0 | 1.000 | 1.0000 [0.9983, 1.0023] |
| (2,4,256,512) | (-1, 1) | 0.9999994 | 1.94e-3 | 1.56e-4 | 4.85e-4 | 0 | 1.000 | 1.0000 [0.9983, 1.0023] |
| (1,1,1024,2048) | (-1, 1) | 0.9999994 | 1.94e-3 | 1.56e-4 | 4.85e-4 | 0 | 1.000 | 1.0000 [0.9983, 1.0023] |
| (1,1,32,32) | (-0.3, 0.7) | 0.9999984 | 1.72e-3 | 9.52e-4 | 3.60e-3 | 1 | 0.193 | 0.9965 [0.9928, 1.0012] |
| (1,1,1024,2048) | (-0.3, 0.7) | 0.9999983 | 1.72e-3 | 9.38e-4 | 3.61e-3 | 1 | 0.197 | 0.9965 [0.9928, 1.0014] |

With bounds (-4, 4) (mostly unclamped) on every shape: PCC 0.9999987, max abs 3.1e-2 (on values near 16, which is
0 ULP), 0 ULP, exact fraction 1.000.

**Assessment.**

- With bf16-representable bounds the op is **bit-exact** against the bf16-rounded fp32 reference (0 ULP on every
  element). The abs/RMS error shown is only the reference's own bf16 output rounding.
- With non-bf16-exact bounds the error is ≤ 1 ULP. The ratio median of 0.9965 is **not** a scale bug. The
  clamp-pinned elements dominate the median: with `randn·2`, most elements exceed 0.7. For those elements the
  clamped value is held in 16-bit DEST, so `0.7 → 0.69921875` and `0.69921875² / 0.49 = 0.9978`, which is then
  rounded. This matches what `torch.clamp` on a bf16 tensor produces, because the bound becomes bf16 there too.
- The p5/p95 spread brackets 1.0. No structural or scale defect.

**Recommended tolerances (bf16):** PCC ≥ 0.9999, ≤ 1 bf16 ULP, `rtol=1e-2, atol=2e-3`.

## Verifier CLI Summary

Not run, because there is no golden suite (see the blocker above). Hand-checked equivalent on this board:

- Acceptance: **48/48 pass**. This covers 9 shapes × 4 bound sets, plus keyword-argument, wide-bound and
  equal-bound cases, and 8 rejection cases.
- Precision baseline: 12/12 pass.
- Extended (verifier-added): 8/8 pass.
- supported_fail: 0 (on the acceptance surface). xpass_drift and xfail_wrong_mode: n/a. Every rejection test raises
  the mandated type.

## Performance baseline (device kernel ns, realtime profiler, best of 5 warm runs)

| Shape | Tiles | Cores | Device ns | Effective DRAM GB/s (2T / t) |
|---|---|---|---|---|
| (1,1,32,32) | 1 | 1 | 1 891 | 2.2 |
| (1,1,256,256) | 64 | 64 | 2 345 | 112 |
| (1,1,32,8192) | 256 | 120 | 4 308 | 243 |
| (2,4,256,512) | 1 024 | 120 | 13 752 | 305 |
| (8,1,1024,1024) | 8 192 | 120 | 87 161 | 385 |
| (1,1,4096,4096) | 16 384 | 120 | 182 535 | 368 |

Large tensors reach about 72–75% of the Blackhole nominal DRAM peak (~512 GB/s), with each core's NoC batch capped at
8 tiles by the DEST cap. Mid-size tensors (≤ ~1k tiles, ≤ 9 tiles per core) are fill- and launch-latency bound. Both
regimes are filed as perf refinements.

## Recommendations

- **Author the golden suite first** (see the blocker). Every `Done when` in `op_requirements.md` uses the
  acceptance + extended tests until then.
- **The acceptance test encodes the Phase-0 rejection boundary.** `test_rejects_wrong_dtype` (float32),
  `test_rejects_row_major_layout`, `test_rejects_non_tile_aligned`, `test_rejects_l1_interleaved` and
  `test_rejects_sharded` assert that values the queue later *adds* are rejected. The refinement that adds each value
  must retarget its rejection test to a still-unsupported value (or retire it) in the same change. This is called
  out per refinement.
- **Precision (no lever needed).** If a future spec wants bounds honored at fp32 rather than bf16 precision on clamped
  elements, `fp32_dest_acc_en=True` (part of Refinement 4's config surface) is the lever. Today the 1-ULP effect
  matches torch-on-bf16 semantics.
- **SFPU vs FPU square** (design perf lamp). It is not worth measuring while the op is DRAM-bound (≥ 300 GB/s). Revisit
  only if a profile shows compute-bound cores, e.g. after the bf8b dtype shrinks the bytes.
