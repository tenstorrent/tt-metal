# Verification Report: multigammaln

## Code Review

### Fixes Applied

1. **`UnpackToDestMode::UnpackToDestFp32` was not set on the fp32 CBs.** The
   program descriptor declared `cb_input_tiles`, `cb_output_tiles`, and the four
   `cb_lgamma_*` intermediates as `Float32`, and the compute kernel ran with
   `fp32_dest_acc_en=True`, but the unpack-to-DEST mode defaulted to the
   SrcA/SrcB path which TF32-truncates intermediate reloads. Symptom: inputs
   within ε of `0.5` produced `+inf` in sub-phase B where torch returns finite
   values (the kernel author had documented this as a "known LLK-level
   limitation" — it was actually a missing precision-config flag).

   **Fix** (`multigammaln_program_descriptor.py:165–185`): build a per-CB
   `unpack_to_dest_mode` vector, set every fp32 CB used by this op to
   `UnpackToDestFp32`, and attach the vector via
   `compute_config.unpack_to_dest_mode = unpack_modes`.

   **Result**: `test_multigammaln_out_of_domain_produces_nan` now passes
   (previously failed with `max abs diff = inf` for inputs `a ≈ 0.500`).

2. **Kernel header comment was stale.** It described the `+inf` near 0.5 as a
   known limitation. Replaced with a precision note explaining that
   `UnpackToDestFp32` is required for that path to deliver finite results, and
   why.

### Design Conformance

| Dimension | Conformant? | Notes |
|-----------|------------|-------|
| Algorithm (4× fp32 lgamma + sum + constant) | ✓ | Matches the design exactly — four `lgamma_with_offset<offset>()` invocations per input tile, each replicating `lgamma_kernel.cpp` plus a `sub_unary_tile(offset)` pre-step. |
| Data pipeline topology | ✓ | reader (NCRISC) → `cb_input_tiles` → compute (TRISCs, 4× sub-phase A → sub-phase B) → `cb_output_tiles` → writer (BRISC). All five CBs configured as designed. |
| Parallelization | ✓ | One output tile per work unit, `split_work_to_cores`, two-group split, per-core RT args populated by walking `core_group_1` then `core_group_2`. |
| Inter-core communication | ✓ | None required — embarrassingly parallel. |

### Deviations from Design (deferred, with justification)

- **Sub-phase B uses raw APIs, not `sfpu_chain + sfpu_pipeline`.** The design
  prescribed expressing the 4-way sum + constant as
  `sfpu_chain(Load×4, SfpuAdd×3, AddScalar)`. Empirically, this does not work:
  `sfpu_chain` in `sfpu_helpers.hpp:1363–1371` returns a
  *default-constructed* chain from the type list, discarding every passed-in
  op instance. `AddScalar<>{... , scalar = K}` therefore loses its `K` and adds
  zero (verified empirically — output was off by exactly `3·log(π)`). This is a
  limitation of the kernel-lib framework, not of this op. Listed as
  Refinement 4 in `op_requirements.md`.

### Helper Usage (post-fix)

- Sub-phase A: necessarily raw APIs (no helper covers
  `lgamma_stirling_float_tile` + `lgamma_adjusted_tile`; the existing `Lgamma`
  helper resolves to the bf16 single-arg form per
  `op_design.md` "Helpers considered and rejected"). Refactor would need a new
  helper in `sfpu_helpers.hpp`.
- Sub-phase B: blocked by the chain-value-preservation issue noted above.

### Correctness Checks

| Check | Status |
|-------|--------|
| TensorAccessor (not deprecated InterleavedAddrGen) | ✓ |
| `void kernel_main()` form, not deprecated namespace pattern | ✓ |
| Include paths use `api/dataflow/dataflow_api.h`, `api/compute/...` | ✓ |
| CB sync: push count = pop count for every CB across one outer iteration | ✓ (verified by reading kernel: 1 push, 1 pop per CB per tile) |
| `cb_input_tiles` held across the four sub-phase A calls, popped exactly once at the end | ✓ |
| Pack/copy DEST slot bookkeeping inside each sub-phase A and sub-phase B block | ✓ — `tile_regs_acquire/commit/wait/release` matched at every site |

## Precision Baseline

Measured with `test_multigammaln_precision_baseline.py`, in-domain inputs
`a ∈ [1.6, 6.6]`, fp32 inputs, fp32 outputs:

| Shape | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err |
|-------|-----|-------------|--------------|------------------|
| (1, 1, 32, 32)    | 0.9999993 | 0.0527 | 3.86e-3 | 6.23e-4 |
| (1, 1, 32, 256)   | 0.9999994 | 0.0531 | 3.80e-3 | 6.08e-4 |
| (1, 1, 256, 32)   | 0.9999994 | 0.0531 | 3.80e-3 | 6.08e-4 |
| (2, 4, 64, 128)   | 0.9999993 | 0.0539 | 3.88e-3 | 6.31e-4 |

**Assessment**: errors are consistent with the LLK-level sources catalogued in
`numerical_stability.md` — primarily the approximated reciprocal in
Stirling's Bernoulli correction and the `reflection_adj − res_stirling`
catastrophic cancellation in the lgamma reflection branch. The fp32 mantissa
survives the sub-phase B reload (now that `UnpackToDestFp32` is configured), so
the 4-way sum does not amplify error further. Output magnitudes for these
shapes are typically 5–15, so a max abs of ~0.05 is roughly 0.5% relative —
identical to what a single in-isolation `ttnn.lgamma` call would produce.

**Recommended Phase-0 tolerances** (mirrored in
`test_multigammaln_precision_baseline.py` and in `op_requirements.md`):
- `PCC >= 0.999` (Phase-0; routine ~0.9999993 today)
- `rtol = 0.05`, `atol = 0.2` (multi-step compute band, set by the acceptance test)

## Test Results

After all fixes:

| Test file | Pass / Total |
|-----------|--------------|
| `test_multigammaln.py` (acceptance — immutable) | 18 / 18 |
| `test_multigammaln_precision_baseline.py` | 4 / 4 |
| `test_multigammaln_extended.py` | 5 / 5 |
| **Total** | **27 / 27** |

Notable cases newly passing after the `UnpackToDestFp32` fix:
- `test_multigammaln_out_of_domain_produces_nan` — previously failed because
  inputs at `a ≈ 0.500` produced `+inf` instead of finite values.
- `test_multigammaln_domain_boundary_strip` (extended) — covers the entire
  strip `a ∈ (1.6, 2.5)` which exercises the reflection branch densely; would
  have been at risk under the prior TF32 reload path.

## Recommendations

Bringing in findings from `data_transfer.md` and `numerical_stability.md`:

1. **Compute config exposure (Refinement 1)** — every important precision lever
   today is hard-coded: `math_fidelity`, `fp32_dest_acc_en`, and now the per-CB
   `unpack_to_dest_mode` vector. Callers cannot trade precision for performance.
   The numerical-stability analyzer notes that `HiFi4` is functionally inert in
   this kernel (there is no FPU multiply path active) — exposing `math_fidelity`
   would let a caller run LoFi at no precision cost. `fp32_dest_acc_en` is the
   real lever and should be exposed last.

2. **Non-tile-aligned shapes (Refinement 2)** — the validator rejects any
   `H % 32 != 0` or `W % 32 != 0`. Adding a padding path with a final mask
   would let callers use natural shapes. The reader and kernel both assume
   tile alignment (reader sticks size = `tile_size(fp32)`); a padding refinement
   needs to teach both to handle the last tile specially.

3. **Variable order `p` (Refinement 3)** — `p` is currently pinned to 4 by the
   four `lgamma_with_offset<>` calls and the baked-in constant
   `THREE_LOG_PI_BITS`. Generalizing requires templating the kernel over a
   compile-time `p` (one CB and one sub-phase A call per term) and computing
   the constant at the program-descriptor level.

4. **Helper-framework limitation — scalar ops in chains (Refinement 4 /
   upstream)** — `sfpu_chain` discards scalar member values. Until the
   framework forwards op instances (e.g., by changing `sfpu_chain` to return
   `ChainFromList<Compacted>::type{ops...}` for the value-preserving
   constructor), `AddScalar` / `MulScalar` / `Rsub` / `Fmod` cannot be used in
   chains. This is an upstream fix; we cannot land it as part of multigammaln.

5. **Refinement priority ordering** (matches `op_requirements.md`):
   compute-config exposure → padding → variable `p` → upstream helper fix. The
   precision baseline already exceeds the design target, so precision tuning
   ranks below shape/feature expansion.

## Files Touched

- `ttnn/ttnn/operations/multigammaln/multigammaln_program_descriptor.py` — added
  per-CB `unpack_to_dest_mode` vector.
- `ttnn/ttnn/operations/multigammaln/kernels/multigammaln_compute.cpp` —
  refactored sub-phase B into a `sum_and_add_const()` helper (cleaner, easier
  to fold into a refinement once the chain framework supports scalars); updated
  the header precision comment.
- `tests/ttnn/unit_tests/operations/multigammaln/test_multigammaln_precision_baseline.py` —
  new precision baseline test.
- `tests/ttnn/unit_tests/operations/multigammaln/test_multigammaln_extended.py` —
  new focused extended-coverage tests (L1 output memory, large magnitudes,
  domain-boundary strip, multi-core stress).
- `ttnn/ttnn/operations/multigammaln/capabilities.md` (new).
- `ttnn/ttnn/operations/multigammaln/op_requirements.md` (new).
- `ttnn/ttnn/operations/multigammaln/changelog.md` (new).
- `ttnn/ttnn/operations/multigammaln/data_transfer.md` (new).
- `ttnn/ttnn/operations/multigammaln/numerical_stability.md` (new — produced by
  the `numerical-stability-analyzer` subagent).
- `ttnn/ttnn/operations/multigammaln/verification_report.md` (this file).
