# Verification Report: moe_down_proj

Scope: Phase 0 as committed by the implementer (R1 build + perf pass, commits 9b751b96e36, c4fe3cdcb75,
06e1e9af592). Board: Blackhole p100a, 110 Tensix cores.

## Code Review

Fixed in place:
- **DRY, CB sizing** (`moe_down_proj_program_descriptor.py`): CB page counts were stated twice, once in
  `footprint_bytes` (as `(X_DEPTH + 1) * ...`, with the staging CBs folded in as an implicit "+1") and again in the
  `_cb(...)` list. Both now read one table, `cb_plan()`, so a knob change (depth, block, staging) lands in one place
  and the budget solve can no longer drift from the allocation.
- **Raw LLK in compute** (`moe_down_proj_compute.cpp`): `PACK((llk_pack_reconfig_l1_acc(0)))` replaced with the
  public compute API `pack_reconfig_l1_acc(0)` (`api/compute/pack.h`).
- **Writer CT-arg order** (`moe_down_proj_writer.cpp`): args 14 and 15 were declared before args 3-13. Reordered to
  match the host list. No functional change.
- **Missing design-required test**: `op_design.md` requires regime-pinned tests for both layouts. Added
  `test_moe_down_proj_layouts.py` (rect and L forced through `layout_override`, at t = 1, 37, 1858, 4096 layer 3,
  one_expert, zeros_between): 12/12 pass.
- **Ledger currency**: see the L1 Ledger Audit below.

Reviewed, no change needed:
- Prompt rules: one `ttnn.generic_op` per call. `expert_counts` is never read on the host (`validate()` reads only
  shape, dtype, layout and memory config). LoFi + fp32 DEST default, a caller config is honored (HiFi2 test), and
  `fp32_dest_acc_en=False` is refused, not overridden. Output writes are bounded by the device clamp
  `active_row_tiles <= tensor_row_tiles`. No host-side tensor transforms.
- Helper usage: `matmul_block` (with the additive `out_base_offset` extension, reviewed: default 0 and asserted 0
  unless `caller_owns_pack_target`), `untilize` (defaults = `InitAndUninit, WaitBlock, UnpackAndPackReconfigure` as
  designed), `ttnn.Mcast` / `McastArgs` for all four channels, `TensorAccessor` for every DRAM / L1 access.
  `void kernel_main()` and `api/...` include paths throughout.
- CB sync: x and w push and pop the nominal round in every step, partials push and pop the nominal block per wave,
  and the counts CBs are pushed once and popped once at kernel end. Balanced on every core, including empty m-groups.
- Blocking-model fidelity: `k_block_tiles`, `num_waves`, `core_row_tiles_max`, `core_n_tiles`, `X/W/OUT_DEPTH` and
  `PREFETCH_ISSUE_STEPS` are host parameters passed as CT args, and the kernels derive every dependent count from
  them. No CB scales with a whole-op dimension: `cb_partials` scales with `core_row_tiles_max`, which the budget
  solve bounds via `num_waves`. The schedule is block-expressed: one `matmul_block` per expert sub-block per round,
  one barrier + one multicast per channel step, one flush per untilized tile row.
- `NUM_EXPERTS` (host) and `MOE_NUM_EXPERTS` (kernel plan header) are both 8. This is a model constant gated by
  `validate()`, not a block knob, so it is left as is.

Design deviations (implementer, measured, accepted):
- w channel moved from the reader RISC to the writer RISC, with w on NoC0 and x on NoC1 (lamp L5; 229 -> 176 us at
  t = 4096).
- One w multicast per K round covering the wave's expert slot range instead of one per expert (lamp L3).
  `compute`'s per-slot cumulative `wait_front` is kept but no longer buys overlap.
- Sender-private staging CBs (`cb_x_stage`, `cb_w_stage`) with a per-step issue quota (lamp L4).
- `CB_BUDGET_BYTES` raised from 800 to 960 KiB against y's measured lowest L1 address at R = 8416.

## Registry Conformance

- `INPUT_TAGGERS = {}`, `SUPPORTED` (6 axes, matching `feature_spec.TARGET`'s axes), `EXCLUSIONS = []`, and
  `validate()` (structural `ValueError` -> model-constant `UnsupportedAxisValue` -> SUPPORTED per axis -> EXCLUSIONS)
  are present and wired. The entry point calls `validate()` first. The op file declares no `INVALID`.
- No SUPPORTED auto-fixes were needed (no XPASS drift).
- INVALID audit (`feature_spec.py`): one entry, `{dtype: bfloat8_b, layout: ROW_MAJOR}`. It is the canonical bf8b +
  RM entry, single-tensor, and a structural impossibility. Inert under the current TARGET (layout is TILE only). No
  cross-tensor couplings. Not norm-like, so no weight canonicalization applies.

## L1 Ledger Audit

- **Currency**: 8 CBs declared, 8 rows. Size expressions match `cb_plan()`. Fixed: `cb_w_round`'s producer was
  listed as the reader, but it is the writer RISC (w channel). Added the staging-CB row to the footprint-terms table
  and a pointer to `cb_plan()` as the host source of truth.
- **Capacity vs live set**: `cb_x_round` (nominal `core_row_tiles_max`) and `cb_w_round` (8 nominal slots, only the
  wave's expert range filled) exceed their live sets by design. Multicast landing addresses must match on every
  core, and nominal pushes keep the ring aligned. Both are justified in the ledger. No collapsed extents: every
  spanned axis has capacity scaling with its knob.
- **Page format vs DEST**: `cb_partials` is Float32 with `fp32_dest_acc_en = True` (the only supported value):
  correct. `cb_output_rows` is Float16_b, the output ABI. x / w are bfp8_b dataflow-only payloads. No finding.
- **Disjoint lifetimes**: none. The staging CBs overlap the round CBs by construction (decoupled sender lookahead),
  and each row justifies this.
- **Bounds / closed form**: every symbol is bounded in the symbol table (`tensor_row_tiles <= 263` via `validate()`,
  `num_waves <= 4`, `k_block_tiles <= 8`), and the total is closed-form.
- **Block-size defaults**: interleaved split across 108 / 110 cores. K block = `K_BLOCK_MAX` = 8, the coarsest that
  fits. The implementer measured that K > 8 lengthens the round-0 fill and loses. `num_waves` = 2 at t >= 3072 is
  forced by the budget at K = 8.
- **Footprint**: `3 * core_row_tiles_max * k_block_tiles * 1088 + 3 * 8 * core_n_tiles * k_block_tiles * 1088 +
  core_row_tiles_max * core_n_tiles * 4096 + 2 * core_n_tiles * 2048 + 64`. That is 940,608 B at t = 4096 (L
  layout). The x terms scale with `core_row_tiles_max x k_block_tiles`, w with `core_n_tiles x k_block_tiles`, and
  partials with `core_row_tiles_max x core_n_tiles`.
- **Data-movement budget**: present and consistent with the code (w 1 DRAM crossing + one straddling expert at 2
  waves, x read once per m-group sender). The cheapest-traffic split (R2, range-pruned w) is a `deferred` row whose
  reason ("receive bytes under ejection capacity") was a prediction. The implementer's ablation (sync + multicast
  floor 130 us of 173 us at t = 4096) contradicts that premise. Folded into perf Refinement 2, and noted in the
  ledger.

## Precision Baseline

`test_moe_down_proj_precision_baseline.py`, golden input distribution (x = bfp8(gelu_tanh(N(-2.8, 1.7^2))),
w = bfp8(N(0, 0.05^2))), real rows only, fp64 reference on the dequantized operands. Production placement.

Default config (LoFi, fp32 DEST):

| Shape (case) | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | bf16 ULP p50 / p99 | got/true p5 / p50 / p95 |
|--------------|-----|-------------|--------------|------------------|--------------------|-------------------------|
| t37 (R = 320) | 0.999910 | 0.189 | 8.32e-3 | 1.890e-2 | 3.4 / 151 | 0.9750 / 0.9866 / 0.9978 |
| t1024 scaled (R = 2272) | 0.999910 | 0.234 | 8.49e-3 | 1.900e-2 | 3.4 / 155 | 0.9750 / 0.9865 / 0.9976 |
| t4096 layer 0 (R = 8416) | 0.999910 | 0.279 | 8.47e-3 | 1.901e-2 | 3.4 / 153 | 0.9761 / 0.9865 / 0.9964 |
| t4096 layer 3 | 0.999910 | 0.278 | 8.47e-3 | 1.901e-2 | 3.4 / 152 | 0.9759 / 0.9865 / 0.9964 |
| t4096 layer 5 | 0.999910 | 0.278 | 8.46e-3 | 1.900e-2 | 3.4 / 152 | 0.9755 / 0.9865 / 0.9968 |

HiFi2, fp32 DEST (caller config):

| Shape (case) | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | bf16 ULP p50 / p99 | got/true p5 / p50 / p95 |
|--------------|-----|-------------|--------------|------------------|--------------------|-------------------------|
| t37 | 0.999999 | 0.022 | 6.45e-4 | 1.683e-3 | 0.3 / 2.6 | 0.9971 / 1.0000 / 1.0029 |
| t1024 scaled | 0.999999 | 0.034 | 6.51e-4 | 1.675e-3 | 0.3 / 2.6 | 0.9971 / 1.0000 / 1.0029 |
| t4096 layer 0 | 0.999999 | 0.032 | 6.50e-4 | 1.675e-3 | 0.3 / 2.5 | 0.9974 / 1.0000 / 1.0026 |

**Assessment**: the LoFi error is the fidelity floor, not a kernel defect. The got/true ratio clusters tightly at
0.9865, the signature of a uniform scale error, so it was triaged. At HiFi2 / HiFi4 (identical) the median moves to
1.0000 and rel RMS drops 11x, so the 1.35% low bias is LoFi's mantissa truncation of the SrcA operand (w). The
kernel's scale is correct. The LoFi numbers match the prompt's stock figures (PCC 0.99991, rel RMS 1.90e-2). The test
therefore runs the scale-bug check at HiFi2 (|p50 - 1| < 1e-3) and only a loose bound at LoFi (< 3e-2).

**Recommended tolerances**: PCC >= 0.9998, rel RMS <= 2.0e-2 (the contract). LoFi rel RMS sits at 1.90e-2, leaving
5% headroom. Any refinement that reorders the K summation or changes the partial / untilize formats must be gated
on this baseline.

## Verifier CLI Summary

Golden run `eval/eval_test_runner.sh eval/golden_tests/moe_down_proj/`, after the code-review fixes: 39 / 93 passed,
0 failed, 0 hangs. `verify_supported` (`verifier_report.json`, copied into the op dir):

- supported_pass: 27
- xfail_expected: 54 (all `buffer_type=DRAM` and/or `out_buffer_type=DRAM`; queued as Refinement 1)
- invalid_skipped: 0 (the only INVALID entry is outside TARGET)
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0
- no_axes_found: 12 (`test_regression.py`, all passing)

Unit dir `tests/ttnn/unit_tests/operations/moe_down_proj/`: 58 passed, 6 skipped (`test_dram_placement`, outside
SUPPORTED). It must run with `--no-precompile`: the precompile collect pass stubs `torch.randn` and the acceptance
test's `lru_cache` keeps the stubbed weight (implementer note).

Device kernel time (`test_moe_down_proj_perf.py --profile`, us, stock / target in parentheses): t4096 L0 173.2
(179.0 / 90.8), L1 174.5 (174.8), L2 175.0 (191.9), L3 178.9 (187.3), L4 173.7 (188.4), L5 173.1 (180.2);
t3072 140.0 (158.9 / 74.6); t2048 113.7 (136.5 / 74.6); t1024 102.8 (127.7 / 74.6). Every row beats stock.
Layer 1 is within 0.2% of stock.

## TARGET - SUPPORTED coverage

| Axis | Missing value | Disposition |
|------|---------------|-------------|
| buffer_type | DRAM | Refinement 1 |
| out_buffer_type | DRAM | Refinement 1 |

No other gap: dtype, layout, fp32_dest_acc_en and memory_layout are single-valued in TARGET and already supported.

## Recommendations

- **L1 headroom risk (no failing cell yet)**: the static CB region ends at about 1,052,640 B. Program creation throws
  `Statically allocated circular buffers ... clash with L1 buffers` if any L1 buffer sits below that address. It was
  reproduced here when a failed test's traceback pinned an earlier call's t = 4096 tensors in L1. In the production
  forward, at R = 8416, x and y already fill L1 from the top, so any other L1-resident activation alive at the call
  will clash. The budget is a constant, not derived from free L1. Deriving it from live allocator state would make
  the program depend on state the program cache does not key on. The placement-aware budget in Refinement 1 is the
  shape-derived part of the fix. If the model integration hits this, lower `CB_BUDGET_BYTES` (three waves cost one
  more straddling expert's w read).
- **Thin stock margin on layer 1** (174.5 vs 174.8 us). Refinement 2 targets the binding multicast scaffold.
  Until then, treat any perf regression on the t = 4096 rows as a contract break.
- **Environment**: `tests.ttnn.utils_for_testing` (`assert_with_pcc`) and `models.common.utility_functions` do not
  import in this checkout. `tt_py_test_utils_common` resolves to a sibling `../tt-metal` tree that references
  `ttnn.MX_DTYPES`. The baseline test uses `models.common.metrics.compute_pcc` / `comp_allclose` instead. Tracy
  writes its CSV under `../tt-metal/generated/profiler/reports/`.
