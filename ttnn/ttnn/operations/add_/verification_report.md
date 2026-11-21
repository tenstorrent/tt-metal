# Verification Report: add_

Target: Quasar (`quasar`) on the ttsim simulator (no clock: perf here is structural, see the backend page). Specification: `eval/specs/add_/spec.py` (sha256 7b592bf81373). Regime built: `sharded_resident`.

## Code Review

**Fixed:**
- `add_.py`: the `compute_kernel_config or default` resolution was written out twice (in `validate()` and in `add_()`), and `add_()` recomputed `canonical_activations`. Both now go through one place:
  - `_resolve_config()`, which resolves only through `default_compute_kernel_config()` (R2);
  - the axes dict that `validate()` returns, from which `add_()` now takes the activation.
  The behaviour is unchanged and the R2 lint still passes.

**Checked and found conformant (no change needed):**
- **Specification rules.**
  - R2: one exported `default_compute_kernel_config()`.
  - R3: `validate()` reads neither `math_fidelity` nor `math_approx_mode`; they pass through to `ComputeHardwareConfig`, and fidelity is always set.
  - R4: returns the `input_tensor_a` object, and `a` is the last `generic_op` tensor.
  - R5: `b` is borrowed by a ring with no byte-writing producer.
  - R6: packer `PackRelu::Zero`; the value is the sum rounded once, then clamped.
  - R7: dtype, layout and `eval.memory.memory_axes` placement are compared across both operands, and an identical shard spec is required.
  - R8: per-core tile counts are clamped; empty shards get no kernels; tested on the ResNet-50 grids.
  - R9: `UnsupportedAxisValue` / `ExcludedCell`.
  - X1: `ValueError`.
  - R1 does not apply yet (float32 is not in SUPPORTED); Refinement 2 carries the EXCLUSIONS entry.
- **GUIDANCE.** All followed: in place; no temporary; resident shards used as the rings (no NoC); ReLU in the packer; walks the tensor's pages, not the grid's capacity.
- **Kernels.**
  - `void kernel_main()`, `api/dataflow/...` includes.
  - The kernels bind no TensorAccessor (borrowed rings, zero-copy), so the local shard is consumed natively, not re-read as if it were interleaved.
  - Push = wait per ring: per NEO share, `core_tiles/C + (t < core_tiles%C)`, the same formula in all three kernels, skipping only trailing zero shares.
  - The DM kernels issue no NoC transfer, so no transaction tag is ever set (the one-session test confirms it).
- **Helper usage.**
  - Compute: one `compute_kernel_lib::add` per NEO over its whole share (`PerBlockSize`, `block_size(neo_block_tiles)`); there is no fusable second phase.
  - Reader/writer: credits only, using the raw DFB API. No helper credits a resident ring (op_design.md lists the helpers considered).
  - Broadcast: `BroadcastDim::None`, both operands full tiles; there is nothing redundant to fill.
- **Blocking-model fidelity.**
  - `block_row_tiles` is the whole resident shard (the sharded default). `block_col_tiles` is the full padded width. `neo_block_tiles = shard_tiles // C`, defined once on the host and passed as a CT arg.
  - The per-thread share is derived in-kernel from the RT `core_tiles` and the CT `compute_threads`. No literal is duplicated, and no ring is sized to an op dimension other than the borrowed shard (the sanctioned resident path).
  - Expression: the reader credits a whole share per call; compute waits/pops/reserves/pushes once per block; the writer acknowledges a whole share. There is no per-tile handshake in any kernel.
- **Thread plan (read off the host program).**
  - `THREAD_PLAN = (1, 4, 1)` is one named constant. `clamp_thread_plan` brings C down to divide the shard's tile count and to at most the busiest core's tiles; R and W = gcd with C.
  - The counts reach every kernel as CT args, and the loops are sized from them and `get_my_thread_id()`.
  - The design states the op as compute-bound in this regime (0 B move), which justifies C = 4; R = W = 1 because the DM harts only post credits.
  - Present and passing: every-thread-ran (`tag_threads`, odd NEOs subtract); the helper at C = 1 / 2 / 4; and `add_` after another program and before `ttnn.to_memory_config` + `ttnn.to_layout`, in one session.
  - The backend has no timing model, so the plan was not measured against (1, 1, 1) / (1, 2, 1). The design's perf-lamp row records that it should be measured on `ttsim-perf`.
- **Work distribution.** The grid is the caller's shard grid, as the design specifies for this regime. Re-spreading over idle clusters is a deferred regime with a positive reason (+3 × tensor bytes over the NoC against 0 B now).

**Deferred to refinements (architectural, not fixable in this pass):** none of the findings. The open TARGET values are queued in `op_requirements.md`.

**Advisories:**
- The C clamp falls to C = 1 on shards with an odd tile count (e.g. 1 tile row × 3 tile columns). The STRIDED ring needs `C | num_entries`, and a borrowed ring must cover the whole shard. No golden or traced shape hits it (the ResNet widths are 8/16/32/64 tiles), but it caps NEO use on odd-width shards.
- `DataFormatReconfig::Enabled` on the chain output is redundant while all three rings share one format. It is harmless, so it was left as is; Refinement 2 must set it `Disabled` on any `UnpackToDest` chain.

## Registry Conformance

- `INPUT_TAGGERS = AXES` (imported from the specification); the op defines no axis function. `SUPPORTED` equals `PHASE0` exactly; `validate_contract --phase0`: PASSED, 0 errors (5 UNRESOLVED open items, all the specification's). `EXCLUSIONS = []`. `validate()` gates SUPPORTED per axis, then EXCLUSIONS, then the regime's shard-geometry checks. `add_()` calls `validate()` first.
- The op file does not declare `INVALID`.
- No SUPPORTED auto-fixes were needed (xpass_drift = 0).
- INVALID audit (specification): one entry, `{memory_config: DRAM, sharding: height}`. Both axes describe the one placement of the call (both operands alike), and it is structural (the runner builds no DRAM-sharded tensor; CONVENTIONS §2.3). It passes the three sanity rules. The bf8b+ROW_MAJOR entry does not apply: neither `bfloat8_b` nor ROW_MAJOR is in TARGET. There are no weights, so no norm-like canonicalization applies. No specification defect.

## L1 Ledger Audit

- **Ledger currency.** It matches the code: three rings, all `dfb_spec_from_sharded_tensor(..., borrowed_from=…)` with `num_entries` = pages per shard (`cb_a_shard`←a, `cb_b_shard`←b, `cb_result_shard`←a), and the size expressions agree.
- **Capacity vs live set.** Capacity `shard_tiles` ≥ live set `core_tiles_k`. The gap appears only on partly filled or empty shards (R8), it has a mechanism reason (a borrowed ring cannot wrap at an offset), and it costs 0 op-owned bytes. On the under side, every axis is spanned and every capacity scales with the shard. No finding.
- **Page format vs DEST.** bf16 pages with `fp32_dest_acc_en` on. This is answered by an output-ABI reason (the page is the caller's tensor `a`, rounded once at store per DEFINITION/R6); no later phase reads a truncated accumulation. No finding.
- **Disjoint lifetimes.** None: all three rings are live for the whole program. `cb_a_shard`/`cb_result_shard` sharing bytes is justified (an in-place transform, two rings because one ring has one producer and one consumer).
- **Bounds.** Every symbol has a bound and a predicate (`shard_tiles ≤ L1/tile_bytes`, `neo_block_tiles ≤ 65,535` uint16 credits). A 1,605,632 B borrowed ring is tested and works (above the target page's 1 MiB per-ring figure; the implementer filed friction).
- **Block-size defaults.** Held. Sharded uses one block, the whole resident shard; per NEO, one helper call over its whole share.
- **Per-core footprint.** `op_owned_L1 = 0` for every knob value. Data-movement budget: DRAM 0 B, NoC 0 B, core-local `a` read + written once and `b` read once. This is the cheapest split possible, and it is implemented.
- **Folded into the queue.** Refinement 1 must add the allocated interleaved rings and their traffic budget to the ledger. No other ledger findings.

## Precision Baseline

`tests/ttnn/unit_tests/operations/add_/test_add__precision_baseline.py`: bfloat16, height-sharded L1 (`auto_shard_config`), RELU, default config (`fp32_dest_acc_en=True`, HiFi4). The reference is the float64 sum of the stored operands, with RELU.

| Shape | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | ULP max / p99 (bf16) | got/true ratio median [p5, p95] |
|-------|-----|-------------|--------------|------------------|----------------------|----------------------------------|
| (1,1,32,64) | 0.9999974 | 0.0156 | 7.53e-4 | 0.00226 | 0 / 0 | 1.0000 [0.9972, 1.0027] |
| (2,4,47,1024) | 0.9999976 | 0.0156 | 7.34e-4 | 0.00218 | 1 / 0 | 1.0000 [0.9972, 1.0028] |
| (1,1,784,512) | 0.9999976 | 0.0156 | 7.32e-4 | 0.00218 | 1 / 0 | 1.0000 [0.9972, 1.0028] |
| (1,1,3136,256) | 0.9999976 | 0.0156 | 7.34e-4 | 0.00218 | 1 / 0 | 1.0000 [0.9972, 1.0028] |

**Assessment**:
- The result is the exact sum rounded once to bfloat16. The ULP p99 is 0 against the float64 reference, here and in all 22 golden passes (`ulp_p99 = 0.0`).
- A few elements are 1 ulp off, consistent with a tie-rounding difference between the packer and torch's round-to-nearest-even. That is inside the specification's 1-ulp grade.
- The absolute and relative-RMS errors are the bf16 quantization of the result itself.
- The ratio is centred at exactly 1.0, with a spread of ±2⁻⁸·⁵ (one bf16 rounding), so there is no scale or structural bug.
- Regression data (golden `test_numeric`): scale 1e-3 and 1e3, all-positive and all-negative all pass; all-negative gives exact zeros, so the packer ReLU is modelled on ttsim.

**Recommended tolerances**: the defaults hold; the specification's resolved numeric entry (bf16 1 ulp) governs.

## Verifier CLI Summary

Final run: `/tmp/add__golden_final`. A copy of `verifier_report.json` is in this directory.

- supported_pass: 22 (14 PHASE0 cells + 8 numeric regression cases)
- xfail_expected: 850
- invalid_skipped: 288
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0
- no_axes_found: 2 (`test_rejection[X1]` passed; `test_op_loose[NOTSET]` skipped, because LOOSE_CASES is empty)
- queue_gaps: 0 (`--queue op_requirements.md`)

Op tests: `scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/add_/` passes 20/20 (acceptance 4, plan 12, precision baseline 4).

## Recommendations

- **Queue.**
  - Refinement 1, interleaved DRAM/L1 regime plus the gate-only axes (activation None, w_non_aligned, ranks 2/3): verifier probe 002 shows the gate-only axes already correct on the sharded path.
  - Refinement 2, float32 plus the 16-bit accumulator, with the R1 exclusion.
  - Refinement 3 (perf, structural gate), a bank-local deal for L1-interleaved operands: verifier probe 003 shows pages dealt round-robin over 32 L1 banks, one per core, with the mapping readable on the host.
  - That leaves two generality refinements, because the TARGET gap is small and the remaining values are gate-only knob-turns that must bundle.
- **Not queued, needs a timed backend: re-spreading narrow shard grids.** ResNet-50's traced calls use 2 of 32 clusters (8 of 128 NEOs). The deferred `respread_narrow_shard_grid` regime would engage the whole grid at the cost of 3 × moved bytes over the NoC, against 0 B now. Whether that wins for one FPU add per tile cannot be judged without a clock. Measure it on `ttsim-perf` (or hardware) before it is queued. The same applies to the thread-plan lamp, (1, 4, 1) against (1, 2, 1) / (1, 1, 1).
- **No other perf headroom in `sharded_resident`.** It moves 0 B, engages all 4 NEOs on every shard whose tile count is divisible by 4, and makes one helper call per NEO; no structural count is left to improve. The DRAM interleaved path is data-movement-bound (plan (1, 1, 1)), so Refinement 1 must simply build it at that plan.
- **Target page friction (the implementer's breadcrumb).** The target page's 1,048,560 B per-ring cap did not bind a borrowed ring: 1.6 MB was accepted and correct.
