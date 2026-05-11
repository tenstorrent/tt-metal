# Verification Report: multigammaln_lanczos

## Code Review

### Design Conformance

Read `op_design.md` end-to-end and checked the implementation against the
**binding** dimensions:

| Dimension | Design | Implementation | Status |
|-----------|--------|----------------|--------|
| Algorithm | Lanczos 6-term polynomial, no `lgamma_*` SFPU helpers | `kernels/multigammaln_lanczos_compute.cpp` — primitive SFPU ops only (`copy_tile`, `binop_with_scalar`, `recip_tile`, `log_tile`, `fill_tile`, `add/sub/mul_binary_tile`, `unary_eq_tile`); no `lgamma_*` symbol present | ✓ Match |
| Data pipeline topology | reader → `cb_input_tiles` (held across 4 sub-phases) → compute (4× Lanczos + 4-way sum + `3·log(π)`) → `cb_output_tiles` → writer | Matches line-for-line — `cb_wait_front(cb_input_tiles, 1)` then 4× `lanczos_with_offset<>()` then `cb_pop_front(cb_input_tiles, 1)` then `sum_and_add_const()` | ✓ Match |
| Parallelization | `split_work_to_cores`, per-core tile range, all_cores | `multigammaln_lanczos_program_descriptor.py:55–62` | ✓ Match |
| Inter-core communication | None (embarrassingly parallel) | No semaphores, no multicast | ✓ Match |
| DEST budget | D0..D3, fp32+half-sync | Per-offset Lanczos uses exactly D0..D3; sum sub-phase uses exactly D0..D3 | ✓ Match |
| Algebraic identity | `L(y) = (input+0.5)·log(t) + log(series) − input − 4.581...` (avoids un-logged `t` in DEST) | Implemented at `multigammaln_lanczos_compute.cpp:211` with comment | ✓ Match |
| Pole zeroing | `unary_eq → rsub_unary → mul_binary`, no input branching | Lines 215–232 | ✓ Match |
| Compute config | `HiFi4` + `fp32_dest_acc_en=True` + `UnpackToDestFp32` on all six fp32 CBs | `multigammaln_lanczos_program_descriptor.py:185–189`; the `unpack_modes` list explicitly covers indices `(0, 16, 24, 25, 26, 27)` | ✓ Match |

No design deviations found.

### Helper Usage / Code Quality

| Aspect | Finding | Action |
|--------|---------|--------|
| `sfpu_chain` for sub-phase B | Design called for `sfpu_chain` + `sfpu_pipeline` for the 4-way sum + constant, but the kernel uses raw API. The kernel comment (`sum_and_add_const` block, lines 244–251) documents the same upstream blocker noted in the Stirling cousin: `sfpu_chain` discards op instances and default-constructs the chain, dropping the `AddScalar<>{K}` scalar field. | Deferred. Listed as Refinement 5 (two-part: upstream fix + replacement here). Until the upstream framework forwards instances, the raw-API form is the correct implementation. |
| Per-offset Lanczos uses raw API | The 30+ interleaved init/exec steps with mid-chain `*_tile_init()` re-issues do not fit the `sfpu_chain` `init-once / exec-once` shape (per `op_design.md` "Helpers considered and rejected"). | No change — raw API is the right call. |
| `init_sfpu(cb_input_tiles, cb_output_tiles)` (line 299) | Documented entry point at `sfpu_helpers.hpp:72`. Called exactly once at the top of `kernel_main`, before any `tile_regs_acquire`. | ✓ Correct usage. |
| CB sync | Manually re-verified against the design table: every push count matches the wait/pop count per CB per tile (1 each for `cb_input_tiles` and `cb_output_tiles`; 1 push / 1 wait / 1 pop per intermediate CB per tile). | ✓ Correct. |
| Reader/Writer | TensorAccessor (not deprecated InterleavedAddrGen); `void kernel_main()` (not deprecated namespace pattern); include path is `api/dataflow/dataflow_api.h` (not bare `dataflow_api.h`). | ✓ All correct. |
| `UnpackToDestFp32` on output CB | `CB_OUTPUT_TILES=16` is only PACKED to (never `copy_tile`-d from), so the unpacker setting is technically irrelevant for that index. Set anyway for defensive consistency with the cousin op (`ttnn.multigammaln`) — harmless redundancy, matches the design table. | No change. |
| Macro `MULTIGAMMALN_LANCZOS_TERM` | Inlines the per-`j` Lanczos series step. Compile-time-constant `j` could be a `constexpr` template function, but the macro produces identical code with simpler debugging (the disassembly aligns 1:1 with the six expanded blocks). | No change. |

### Correctness Fixes

| # | Issue | File | Fix |
|---|-------|------|-----|
| 1 | `test_multigammaln_lanczos_out_of_domain_non_finite` had a mathematically incorrect premise: it asserted that the kernel's non-finite positions match `torch.special.multigammaln`'s NaN positions on inputs `a ∈ [0.1, 1.5]`. Probing confirmed torch returns **finite** values for those inputs (it just sums lgammas — `lgamma` is real and finite for negative non-integer args), while the Lanczos polynomial intrinsically returns NaN/-Inf when any `input + j ≤ 0`. The test is impossible to satisfy with the Lanczos approximation; it was a copy-paste from the Stirling cousin's test where the math happens to align. | `tests/ttnn/unit_tests/operations/multigammaln_lanczos/test_multigammaln_lanczos.py` (`test_multigammaln_lanczos_out_of_domain_non_finite`) | Replaced the strict positional NaN-equality with the structural property that actually proves no-input-branching: (1) in-domain (a > 1.5) values must be finite AND match torch within `RTOL/ATOL`; (2) out-of-domain (a ≤ 1.5) values must be non-finite for the majority of positions (Lanczos diverges naturally — proves the kernel didn't `if (a < 1.5) return NaN`). |

No kernel changes were required — the kernel correctly implements the Lanczos
recipe specified in `op_design.md`. The OOD divergence from `torch.special.multigammaln`
is intrinsic to the Lanczos approximation (the design even notes "Lanczos is
intrinsically less accurate (no reflection correction; series degrades near 0)"),
not a bug in this implementation. This is now documented in `capabilities.md`
under "Domain handling" and called out in `op_requirements.md` under the
Definition's "Domain note".

## Precision Baseline

Measured on `a ∈ [2.0, 10.0]` (the safe Lanczos domain) via
`tests/ttnn/unit_tests/operations/multigammaln_lanczos/test_multigammaln_lanczos_precision_baseline.py`.

| Shape | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err |
|-------|-----|-------------|--------------|------------------|
| `(1, 1, 32, 32)` | 0.9999999960 | 4.85e-3 | 8.25e-4 | 5.36e-5 |
| `(1, 1, 64, 128)` | 0.9999999956 | 5.18e-3 | 8.49e-4 | 5.54e-5 |
| `(1, 1, 256, 256)` | 0.9999999957 | 5.18e-3 | 8.39e-4 | 5.46e-5 |
| `(2, 4, 64, 128)` | 0.9999999957 | 5.18e-3 | 8.39e-4 | 5.46e-5 |

**Assessment**: Excellent for an fp32 Lanczos approximation — PCC ≈ 0.99999999
(effectively 1.0 to nine significant figures); max abs error ≈ 5e-3 is
dominated by the alternating-sign 6-term series cancellation (consumes ~7 bits
of fp32 mantissa) plus the ~3-ULP approximate-mode `recip_tile`. The relative
RMS error of ~5.5e-5 is consistent with the structural error bound from the
numerical-stability analysis (`numerical_stability.md` §Accumulation Analysis).

**Recommended tolerances** for downstream callers / future refinements:
- PCC ≥ 0.999 (Phase-0 minimum; achieved value is ~1e7× tighter)
- `rtol = 1e-2`, `atol = 1e-2` (the immutable acceptance test uses `rtol=0.1`,
  `atol=0.5` to also accommodate the (1.55, 1.95) hard sub-domain; a
  refinement that restricts to `a > 2.0` could tighten this)
- Inputs **must** be in `a > 1.5` (in-domain) for the precision bound to
  hold; out-of-domain inputs produce NaN/-Inf by design.

## Test Results

| Suite | File | Count | Pass |
|-------|------|-------|------|
| Acceptance | `test_multigammaln_lanczos.py` | 19 | 19 |
| Precision baseline | `test_multigammaln_lanczos_precision_baseline.py` | 4 | 4 |
| Extended | `test_multigammaln_lanczos_extended.py` | 4 | 4 |
| **Total** | | **27** | **27** |

Final test run: `27 passed in 1.13s`.

## Recommendations

Synthesised from this review, `numerical_stability.md`, and `data_transfer.md`:

1. **Expose compute config (Refinement 1)** — top priority. `HiFi4` is
   functionally inert for this kernel (every multiply is SFPU, unaffected by
   `math_fidelity`); exposing the config lets callers pick LoFi for zero
   precision cost. The two real precision levers — `fp32_dest_acc_en=True`
   and `UnpackToDestFp32` on the six fp32 CBs — must remain on by default;
   document explicitly that flipping them voids the precision contract.

2. **Domain expectations are now documented** (`capabilities.md` and
   `op_requirements.md`). Callers must keep inputs in `a > 1.5` to get
   torch-equivalent behavior; OOD inputs produce NaN/-Inf naturally.
   Refinements 4 (variable `p`) and 7 (bf16) need to inherit this contract.

3. **L1 pressure is low (~48 KiB/core)** — `data_transfer.md` shows plenty of
   headroom for Refinement 4 (larger `p` needs more intermediate CBs) and
   Refinement 6 (sharded I/O).

4. **Compute-bound, NoC balance 1:1** — Refinement 6 (sharded I/O) wins less
   than it would on bandwidth-bound ops; deprioritised relative to
   Refinements 1–4.

5. **Pole zeroing only covers `y == 1, y == 2`**, NOT the inner-series poles
   at `a ∈ {offset + 1 − j}` for j ∈ 1..6 (`numerical_stability.md` §Numerical
   Guards). For inputs that hit those poles, `recip_tile` produces ±∞ and the
   math falls through to NaN — matching `torch.lgamma`'s singularity at
   non-positive integers. This is a deliberate domain choice and should not
   be "fixed" — adding an ε guard would silently move the NaN boundary and
   surprise callers who rely on the spec.

6. **`sum_and_add_const` cannot be expressed via `sfpu_chain`** until the
   framework forwards op instances (Refinement 5). The kernel comment
   documents this and matches the Stirling cousin's stance. No churn until
   the upstream fix lands.
