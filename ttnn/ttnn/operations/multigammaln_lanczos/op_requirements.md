# Operation Requirements: multigammaln_lanczos

## Definition

- **Formula**: `multigammaln_lanczos(a) = L(a) + L(a − 0.5) + L(a − 1.0) + L(a − 1.5) + 3·log(π)`
  where `L(·)` is the **Lanczos 6-term polynomial** approximation of `lgamma`
  (see `op_design.md` "Algorithm"). The `(p*(p−1)/4)·log(π)` constant for
  `p = 4` is `3·log(π) ≈ 3.434189657547`.
- **Inputs**:

  | Name | Role | Shape pattern | Description |
  |------|------|---------------|-------------|
  | `input_tensor` | per-element argument `a` | rank-4 `(N, C, H, W)`, `H % 32 == 0`, `W % 32 == 0`, on-device | fp32, TILE_LAYOUT |

- **Output**: shape == input; dtype fp32; layout TILE_LAYOUT; memory inherits
  from the `memory_config` kwarg (defaults to DRAM interleaved).
- **Parameters**:

  | Name | Type | Default | Range | Description |
  |------|------|---------|-------|-------------|
  | `memory_config` | `ttnn.MemoryConfig` | `DRAM_MEMORY_CONFIG` | DRAM or L1 interleaved | output memory placement |

  No `p` parameter — the order is pinned to 4. No `compute_kernel_config`
  parameter — fidelity / DEST acc / unpack-to-dest mode are pinned internally.

- **PyTorch Reference**:

  ```python
  def torch_reference(a: torch.Tensor) -> torch.Tensor:
      # Equivalent to torch.special.multigammaln(a.float(), 4) for a > 1.5
      # (the kernel diverges from torch in the OOD region — see Domain below).
      return torch.special.multigammaln(a.float(), 4)
  ```

- **Import Path**: `from ttnn.operations.multigammaln_lanczos import multigammaln_lanczos`
- **Function Signature**:

  ```python
  def multigammaln_lanczos(
      input_tensor: ttnn.Tensor,
      *,
      memory_config: ttnn.MemoryConfig = None,
  ) -> ttnn.Tensor:
  ```

- **Validation** (entry point — `multigammaln_lanczos.py:_validate_input`):
  - dtype must be `ttnn.float32` → `ValueError`
  - layout must be `ttnn.TILE_LAYOUT` → `ValueError`
  - rank must be exactly 4 → `ValueError`
  - `H % 32 == 0` and `W % 32 == 0` → `ValueError`

- **Domain note** (important for refinement scoping): The Lanczos approximation
  diverges from `torch.special.multigammaln` for `a ≤ 1.5`. Torch returns
  finite values there (sum of lgammas, each finite at negative non-integer
  args); the Lanczos polynomial returns NaN/-Inf because the alternating-sign
  series goes negative when any `input + j ≤ 0`. **This is an intrinsic
  limitation of the chosen approximation, not a kernel bug.** No refinement
  can recover torch's OOD behavior without switching to Stirling+reflection
  (which is the cousin `ttnn.multigammaln`).

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
  all six fp32 CBs. Not exposed to the caller.
- **Params**: none beyond `memory_config`.
- **Test shapes**: `(1,1,32,32)`, `(1,1,32,256)`, `(1,1,256,32)`,
  `(1,1,64,128)`, `(1,1,128,64)`, `(2,4,64,128)`, plus the `(1,1,512,512)`
  stress shape and the in-domain strip `a ∈ (1.55, 1.95)` for the hard
  sub-region.
- **Precision achieved**: PCC ≈ 0.99999999, max abs ≈ 5.2e-3, mean abs ≈ 8e-4,
  relative RMS ≈ 5.5e-5 across the baseline shapes (`a ∈ [2.0, 10.0]`).

### [ ] Refinement 1 — Expose compute config

- Accept `compute_kernel_config: ttnn.WormholeComputeKernelConfig | None` (or
  the equivalent generic-op compute-config form) on the entry point.
- Plumb `math_fidelity`, `fp32_dest_acc_en`, `dst_full_sync_en`, and the per-CB
  `unpack_to_dest_mode` vector through the program descriptor.
- Defaults must match the current hard-coded values so existing callers do not
  regress.
- **Verifier notes**:
  - `numerical_stability.md` flags `HiFi4` as functionally inert in this
    kernel — every multiply is SFPU (`mul_unary_tile`, `mul_binary_tile`),
    unaffected by `math_fidelity`. Exposing it lets callers run LoFi at zero
    precision cost.
  - `fp32_dest_acc_en=False` would force the kernel into bf16 DEST, which
    would lose mantissa to the alternating-sign series cancellation. The
    refinement must document that flipping it to False voids the precision
    contract.
  - `UnpackToDestFp32` on the six fp32 CBs is **load-bearing** for the
    tile-boundary round-trip — it bypasses the SrcA/SrcB TF32-truncation
    path on `copy_tile` of intermediate CBs in sub-phase B. Defaults must
    keep `UnpackToDestFp32` for indices `(0, 16, 24, 25, 26, 27)`.

### [ ] Refinement 2 — Non-tile-aligned shapes

- Support `H % 32 != 0` and/or `W % 32 != 0` by padding the last tile (the
  validator currently rejects non-aligned inputs at `multigammaln_lanczos.py:81–84`).
- The output tile-pad region must contain undefined values (don't write garbage
  into user-visible memory).
- **Verifier notes**:
  - Reader assumes `tile_id` corresponds to a full tile via
    `noc_async_read_tile`. The padding path needs either (a) host-side padding
    in `multigammaln_lanczos.py` before launch, or (b) kernel-level edge-tile
    handling via a mask CB. Option (a) is the simpler refinement; option (b)
    is a follow-up.
  - The kernel itself is shape-agnostic past the per-tile interface — no
    changes needed in `multigammaln_lanczos_compute.cpp` for option (a).

### [ ] Refinement 3 — Generalize rank support

- Accept rank 2, 3, 4, and 5+ inputs by reshaping to rank-4 internally and
  reshaping the output back.
- Currently `multigammaln_lanczos.py:76–79` rejects anything that isn't rank-4.
- **Verifier notes**: The kernel is purely elementwise — work distribution is
  by tile id, no per-dimension assumptions. The refinement is host-side only:
  reshape input to `(1, 1, prod_leading, last)` (or similar), pass through the
  kernel, reshape output back.

### [ ] Refinement 4 — Variable order `p`

- Accept a `p: int` kwarg with the contract from
  `torch.special.multigammaln(x, p)`.
- Generalize the kernel to run `p` Lanczos sub-phases with offsets
  `0, 0.5, 1.0, …, (p−1)/2`. Each sub-phase needs its own intermediate CB.
- The compile-time constant becomes `(p*(p−1)/4)·log(π)`; recompute the
  fp32 bit pattern at the program-descriptor level and pass it as a
  compile-time arg.
- **Verifier notes**:
  - DEST budget caveat: the sum sub-phase currently uses D0..D3 (stride 4)
    which fits exactly in `DEST_AUTO_LIMIT = 4`. For `p > 4`, the sum must
    fold incrementally (load 4 → fold → load remaining `p-4` → fold) inside a
    single output DEST cycle, or via an accumulator CB.
  - `p = 1` reduces to plain `lgamma_lanczos(a)`; consider short-circuiting at
    the entry point.
  - The Lanczos polynomial's domain shrinks as `p` grows (the lowest term
    `L(a − (p−1)/2)` has the weakest convergence). Larger `p` → narrower
    in-domain region. Refinement should document the per-`p` safe range.

### [ ] Refinement 5 — Use `sfpu_chain` for sub-phase B (blocked on framework)

- The design called for the 4-way sum + constant to be expressed as a single
  `sfpu_chain(Load×4, SfpuAdd×3, AddScalar)` + `sfpu_pipeline` invocation.
  This is currently impossible because `sfpu_chain` discards op instances and
  default-constructs the chain (`sfpu_helpers.hpp:1363–1371` →
  `ChainFromList<...>::type{}`). The same blocker is documented for the
  Stirling cousin (`multigammaln/kernels/multigammaln_compute.cpp:200–206`).
- **Two-part refinement**:
  1. **Upstream**: extend `sfpu_chain` to forward op instances (e.g.,
     `return ChainFromList<Compacted>::type{ops...}` using the existing
     value-preserving `SfpuChain` constructor).
  2. **In this op**: replace `sum_and_add_const()` with the chain form
     (matches the design exactly).
- **Verifier notes**: Until (1) lands, the raw-API form in
  `sum_and_add_const()` is the correct implementation. Don't churn the kernel
  before the upstream fix.

### [ ] Refinement 6 — Sharded input/output support

- Currently DRAM/L1 interleaved only. Sharded inputs would let callers avoid
  the host→DRAM staging step and read directly from L1.
- The reader would need a sharded-path variant; the compute and writer are
  CB-driven and unchanged.
- **Verifier notes**: Lower priority than Refinements 1–4 — the op is
  compute-bound (see `data_transfer.md`), so sharded I/O wins less than it
  would on bandwidth-bound ops.

### [ ] Refinement 7 — bf16 / bfloat8 input support

- Currently fp32-only. Adding bf16 / bfloat8_b requires:
  - Relaxing the entry-point dtype check.
  - Either disabling `fp32_dest_acc_en` (which voids the precision contract
    — the alternating-sign Lanczos series cancellation needs the full fp32
    mantissa) or unpacking bf16 → fp32 in DEST and packing back. The latter
    is the better path but requires `UnpackToDestFp32` on the input CB to
    work as the format-widening path.
- **Verifier notes**: Depends on Refinement 1 (compute config exposure) — a
  bf16 caller must be able to opt into the slower-but-correct precision mode.
  The Lanczos polynomial's intrinsic precision floor is tighter than bf16's
  mantissa (~7 bits), so bf16 inputs will produce noticeably worse accuracy
  than fp32; the refinement should document the achieved PCC for bf16.

### [ ] Refinement 8 — Memory-pressure stress

- Currently per-core L1 CB footprint is ~48 KiB out of 1.5 MiB — plenty of
  headroom. If future refinements (larger intermediate CBs for batching,
  sharded inputs, or higher `p`) push this up, add a sweep for the maximum
  shape that fits in L1 and document the limit in `capabilities.md`.
- Always last in the refinement order.
