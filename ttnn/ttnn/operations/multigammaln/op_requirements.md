# Operation Requirements: multigammaln

## Definition

- **Formula**: `multigammaln(a) = lgamma(a) + lgamma(a - 0.5) + lgamma(a - 1.0) + lgamma(a - 1.5) + 3*log(π)`
  (the `(p*(p-1)/4) * log(π)` constant for `p = 4`; `3*log(π) ≈ 3.434189657547`).
- **Inputs**:

  | Name | Role | Shape pattern | Description |
  |------|------|---------------|-------------|
  | `input_tensor` | per-element argument | rank-4 `(N, C, H, W)`, `H % 32 == 0`, `W % 32 == 0`, on-device | fp32, TILE_LAYOUT |

- **Output**: shape == input; dtype fp32; layout TILE_LAYOUT; memory inherits
  from the `memory_config` kwarg (defaults to DRAM interleaved).
- **Parameters**:

  | Name | Type | Default | Range | Description |
  |------|------|---------|-------|-------------|
  | `memory_config` | `ttnn.MemoryConfig` | `DRAM_MEMORY_CONFIG` | DRAM or L1 interleaved | output memory placement |

  No `p` parameter — the order is pinned to 4.

- **PyTorch Reference**:

  ```python
  def torch_reference(a: torch.Tensor) -> torch.Tensor:
      # Equivalent to torch.special.multigammaln(a.float(), 4)
      return (
          torch.lgamma(a)
          + torch.lgamma(a - 0.5)
          + torch.lgamma(a - 1.0)
          + torch.lgamma(a - 1.5)
          + 3.0 * math.log(math.pi)
      )
  ```

- **Import Path**: `from ttnn.operations.multigammaln import multigammaln`
- **Function Signature**:

  ```python
  def multigammaln(
      input_tensor: ttnn.Tensor,
      *,
      memory_config: ttnn.MemoryConfig = None,
  ) -> ttnn.Tensor:
  ```

- **Validation** (entry point — `multigammaln.py:_validate_input`):
  - dtype must be `ttnn.float32` → `ValueError`
  - layout must be `ttnn.TILE_LAYOUT` → `ValueError`
  - rank must be exactly 4 → `ValueError`
  - `H % 32 == 0` and `W % 32 == 0` → `ValueError`

## Phases

> **Non-regression rule**: every phase must pass all tests from prior phases.
> **Accuracy**: PCC > 0.999 to pass initially. Agents tighten tolerances and
> note achieved values in `changelog.md`.
> **Checkbox protocol**: agents mark `[x]` once a phase is complete and all
> tests pass.

### [x] Phase 0 — Core Implementation

- **Cores**: multi-core (`split_work_to_cores` over the full compute grid).
- **Dtype**: float32 only.
- **Layout**: TILE_LAYOUT only.
- **Memory**: DRAM or L1 interleaved (input); caller-controlled output
  `memory_config`.
- **Compute config**: hard-coded `math_fidelity=HiFi4`,
  `fp32_dest_acc_en=True`, per-CB `unpack_to_dest_mode=UnpackToDestFp32` for
  all fp32 CBs. Not exposed to the caller.
- **Params**: none beyond `memory_config`.
- **Test shapes**: `(1,1,32,32)`, `(1,1,32,256)`, `(1,1,256,32)`,
  `(1,1,64,128)`, `(1,1,128,64)`, `(2,4,64,128)`, plus the larger
  `(1,1,512,512)` stress shape and the in-domain strip `a ∈ (1.6, 2.5)` for the
  reflection branch.
- **Precision achieved**: PCC ≈ 0.9999993, max abs ≈ 0.054,
  relative RMS ≈ 6.3e-4 across the baseline shapes.

### [ ] Refinement 1 — Expose compute config

- Accept `compute_kernel_config: ttnn.WormholeComputeKernelConfig | None` (or
  the equivalent generic-op compute-config form) on the entry point.
- Plumb `math_fidelity`, `fp32_dest_acc_en`, `dst_full_sync_en`, and the per-CB
  `unpack_to_dest_mode` vector through the program descriptor.
- Defaults must match the current hard-coded values so existing callers do not
  regress.
- **Verifier notes**:
  - The numerical-stability analyzer flagged `HiFi4` as functionally inert in
    this kernel (no FPU multiply path is active); exposing `math_fidelity` lets
    callers run LoFi at zero precision cost.
  - `fp32_dest_acc_en=False` would force the kernel into the bf16 round-trip
    inside `lgamma_adjusted_tile`. The op's tolerances are NOT defined for that
    mode; document that flipping it to False relaxes the precision contract.
  - `UnpackToDestFp32` is **load-bearing** for inputs near integer boundaries
    (a ≈ 0.5+ε). The acceptance test
    `test_multigammaln_out_of_domain_produces_nan` will regress if it is left
    at `Default`. The refinement must keep `UnpackToDestFp32` as the default
    for the fp32 CBs.

### [ ] Refinement 2 — Non-tile-aligned shapes

- Support `H % 32 != 0` and/or `W % 32 != 0` by padding the last tile (the
  validator currently rejects non-aligned inputs at `multigammaln.py:75`).
- The output tile-pad region must contain undefined values (don't write garbage
  into user-visible memory).
- **Verifier notes**:
  - Reader assumes `tile_id` corresponds to a full tile via
    `noc_async_read_tile`. The padding path needs either (a) host-side padding
    in `multigammaln.py` before launch, or (b) kernel-level edge-tile handling
    via a mask CB. Option (a) is the simpler refinement; option (b) is a
    follow-up.
  - The kernel itself is shape-agnostic past the per-tile interface — no
    changes needed in `multigammaln_compute.cpp` for option (a).

### [ ] Refinement 3 — Variable order `p`

- Accept a `p: int` kwarg with the contract from
  `torch.special.multigammaln(x, p)`.
- Generalize the kernel to run `p` lgamma sub-phases with offsets
  `0, 0.5, 1.0, …, (p−1)/2`. Each sub-phase needs its own intermediate CB.
- The compile-time constant becomes `(p*(p−1)/4) * log(π)`; recompute the
  fp32 bit pattern at the program-descriptor level and pass it as a
  compile-time arg.
- **Verifier notes**:
  - DEST budget caveat: the sum sub-phase currently uses D0..D3 (stride 4)
    which fits exactly in `DEST_AUTO_LIMIT = 4`. For `p > 4`, the sum must
    fold incrementally (load 4 → fold → load remaining `p-4` → fold) inside a
    single output DEST cycle, or via an accumulator CB.
  - `p = 1` reduces to plain `lgamma(a)`; consider short-circuiting to
    `ttnn.lgamma` at the entry point in that case.

### [ ] Refinement 4 — Use `sfpu_chain` for sub-phase B (blocked on framework)

- The design called for the 4-way sum + constant to be expressed as a single
  `sfpu_chain(Load×4, SfpuAdd×3, AddScalar)` + `sfpu_pipeline` invocation.
  This is currently impossible because `sfpu_chain` discards op instances and
  default-constructs the chain (`sfpu_helpers.hpp:1363–1371` →
  `ChainFromList<...>::type{}`).
- **Two-part refinement**:
  1. **Upstream**: extend `sfpu_chain` to forward op instances (e.g.,
     `return ChainFromList<Compacted>::type{ops...}` using the existing
     value-preserving `SfpuChain` constructor). Add a regression test in
     `tests/ttnn/unit_tests/operations/eltwise/unary` that exercises
     `AddScalar` inside a chain.
  2. **In this op**: replace `sum_and_add_const()` with the chain form
     (matches the design exactly).
- **Verifier notes**:
  - Until (1) lands, the raw-API form in `sum_and_add_const()` is the correct
    implementation. Don't churn the kernel before the upstream fix.
  - Verifier already drafted the chain form during verification and confirmed
    via test output that `AddScalar`'s scalar field is dropped — see kernel
    comment block above `sum_and_add_const`.

### [ ] Refinement 5 — Sharded input/output support

- Currently DRAM/L1 interleaved only. Sharded inputs would let callers avoid
  the host→DRAM staging step and read directly from L1.
- The reader would need a sharded-path variant; the compute and writer are
  CB-driven and unchanged.
- **Verifier notes**: lower priority than Refinements 1–3 — this op is
  compute-bound (data_transfer.md), so sharded I/O wins less than it would on
  bandwidth-bound ops.

### [ ] Refinement 6 — bf16 / bfloat8 input support

- Currently fp32-only. Adding bf16 / bfloat8_b requires:
  - Relaxing the entry-point dtype check.
  - Either disabling `fp32_dest_acc_en` (which voids the current precision
    contract) or unpacking bf16 → fp32 in DEST and packing back. The latter is
    the better path.
  - Adjusting the bit-cast scalars (the `OFFSET_BITS_*` and `THREE_LOG_PI_BITS`
    are fp32-encoded; they would need to be re-encoded per the input format).
- **Verifier notes**: depends on Refinement 1 (compute config exposure) — a
  bf16 caller must be able to opt into the slower-but-correct precision mode.

### [ ] Refinement 7 — Memory-pressure stress

- Currently per-core L1 CB footprint is ~48 KiB out of 1.5 MiB — plenty of
  headroom. If future refinements (larger intermediate CBs for batching,
  sharded inputs) push this up, add a sweep for the maximum shape that fits in
  L1 and document the limit in `capabilities.md`.
- Always last in the refinement order.
