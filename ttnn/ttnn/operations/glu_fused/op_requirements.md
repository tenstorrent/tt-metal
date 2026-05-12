# Operation Requirements: glu_fused

## Definition

- **Formula**: `out[..., j] = x[..., j] * sigmoid(x[..., j + W/2])` for `j ∈ [0, W/2)`
  — i.e. `torch.nn.functional.glu(x, dim=-1)`.

- **Inputs**:

| Name | Role | Shape pattern | Description |
|------|------|---------------|-------------|
| `input_tensor` | x (gated value + gate) | `(N, C, H, W)` with `W % 64 == 0` and `H % 32 == 0` | float32 TILE_LAYOUT tensor on device. First half along last dim is the value, second half is the gate. |

- **Output**: Shape `(N, C, H, W/2)`, dtype `float32`, layout `TILE_LAYOUT`,
  memory_config inherited from input.

- **Parameters**: (none in Phase 0 beyond the input tensor)

| Name | Type | Default | Range | Description |
|------|------|---------|-------|-------------|
| — | — | — | — | Phase 0 exposes no other parameters. Refinements add `dim`, `compute_kernel_config`, etc. |

- **PyTorch Reference**:

```python
def glu_reference(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.glu(x, dim=-1)
```

- **Import Path**: `from ttnn.operations.glu_fused import glu_fused`

- **Function Signature**:

```python
def glu_fused(input_tensor: ttnn.Tensor) -> ttnn.Tensor: ...
```

- **Validation** (rejected with `ValueError`):
  - `input_tensor` not on device
  - `dtype != float32`
  - `layout != TILE_LAYOUT`
  - `rank != 4`
  - `shape[-1] % 64 != 0` (each half must be tile-aligned)
  - `shape[-2] % 32 != 0`

## Phases

> **Non-regression rule**: Every phase must pass all tests from prior phases.
> **Accuracy**: PCC > 0.99 to pass initially. Agents tighten tolerances and note achieved values in changelog.
> **Checkbox protocol**: Agents mark `[x]` when a phase is complete and all tests pass.

### [x] Phase 0 — Core Implementation
- **Cores**: multi-core via `split_work_to_cores` on the full
  `compute_with_storage_grid_size`.
- **Dtype**: float32 only.
- **Layout**: TILE only.
- **Memory**: DRAM and L1 interleaved (output inherits input). No sharded support.
- **Compute config**: hard-coded `HiFi4 + fp32_dest_acc_en=True +
  UnpackToDestFp32` on input CBs. Not exposed to the caller.
- **Params**: none — single positional/keyword `input_tensor`.
- **Test shapes**: `(1,1,32,64)`, `(1,1,32,128)`, `(1,1,256,64)`,
  `(1,1,32,512)`, `(1,1,64,256)`, `(1,1,256,128)`, `(2,4,64,128)`,
  `(2,2,128,256)` (acceptance) plus precision baseline shapes.

### [ ] Refinement 1 — Compute config exposure
- Add an optional `compute_kernel_config: ttnn.WormholeComputeKernelConfig`
  kwarg to `glu_fused`. When provided, override the hard-coded defaults
  (math_fidelity, fp32_dest_acc_en, math_approx_mode, packer_l1_acc).
- Wire `math_approx_mode` through to the compute kernel: if `True`, use
  `Sigmoid<Approx::Fast, Dst::D1>{}` instead of `Approx::Exact` (template
  arg switch in the chain). This unlocks the fast-approx Schraudolph
  sigmoid as a perf/precision tradeoff.
- Keep current defaults (HiFi4 + fp32_dest_acc_en=True + UnpackToDestFp32)
  when no config is passed — must not regress Phase 0 precision.
- **Note from verifier**: `math_fidelity` has **no effect** on this kernel
  (only multiply is SFPU, not FPU). Expose it for forward compatibility but
  don't waste perf measurement budget on tuning it. See
  `numerical_stability.md` § "Math Fidelity Profile".
- **Tests**: Repeat precision baseline with `math_approx_mode=True` —
  expect a small but measurable PCC drop; record in changelog.

### [ ] Refinement 2 — bfloat16 support
- Relax the dtype gate in `glu_fused.py` to accept `bfloat16` in addition
  to `float32`.
- For bfloat16, switch CB formats to bfloat16 (page size 2048 B) and **do
  not** set `UnpackToDestFp32` on input CBs (would waste perf with no
  precision gain since the input is already bf16).
- Verify the chain still functions — Sigmoid LUT and SfpuMul are
  format-agnostic; the unpack/pack format reconfig handles the dtype switch.
- **Note from verifier**: When bfloat16 is added, the precision baseline
  test will need a separate bf16 tolerance set — bf16 PCC will be ≈ 0.999,
  max_abs in the 0.01 range. Don't tighten the bf16 acceptance bounds.
- **Tests**: Re-run all Phase 0 shapes with `dtype=bfloat16`. Add a
  precision baseline row per shape.

### [ ] Refinement 3 — `dim` parameter + arbitrary rank
- Add `dim` kwarg (default `-1`). For `dim ∈ {-1, -2, ..., -rank}` and
  `dim ∈ {0, 1, ..., rank-1}`, compute the per-axis split offset host-side
  and pass `Wt_half`/stride/etc. as compile-time args.
- Relax the rank == 4 gate to rank >= 1.
- PyTorch parity: `torch.nn.functional.glu` supports any rank and any `dim`
  as long as `shape[dim] % 2 == 0`.
- **Note from verifier**: The reader's tile-id math at
  `glu_fused_reader.cpp:42-45` hard-codes the last-dim split. Generalizing
  to other axes means computing `a_tile_idx` and `b_tile_idx` as functions
  of `out_idx`, the strides, and the split offset along the requested
  axis. The compute kernel does not change.

### [ ] Refinement 4 — Non-tile-aligned shapes
- Drop the `W % 64 == 0` and `H % 32 == 0` requirements (or relax to
  `W % 2 == 0` to match PyTorch).
- For `W` such that `W/2` is not tile-aligned, the split sits inside a
  tile — the reader must either issue partial-tile reads, or the program
  must pad to the next multiple of 64 with masking on the pack side.
- For non-tile-aligned `H`, standard padding/masking applies (this is a
  cross-cutting concern shared with most TTNN ops).
- **Note from verifier**: This refinement is the largest of the list —
  it requires both reader-side and pack-side masking logic. Sigmoid on
  garbage padded values must not poison neighboring outputs (sigmoid is
  pointwise so garbage stays in its lane, but mask the output).

### [ ] Refinement 5 — Sharded memory support
- Add a sharded input/output code path. When the input is height-sharded
  or width-sharded, the reader becomes a no-op (data already in L1) and
  the writer becomes an in-place L1 update or a sharded NoC route.
- **Note from verifier**: The current reader uses `TensorAccessor` which
  handles interleaved DRAM/L1. Sharded inputs need a separate code path
  that accesses the local L1 shard directly (zero-copy CB pattern).
- Refinements 1–4 are pre-requisites — adding sharding to a multi-dtype,
  multi-rank op is much easier than re-doing sharding for each
  dtype/rank addition.

### [ ] Refinement 6 — Memory pressure
- Audit L1 footprint for very large shapes. Current per-core CB footprint
  is `4096 B × (2 + 2 + 2) = 24 KB` — well within budget on any device.
- Verify the per-core tile count scales sensibly when
  `total_output_tiles ≫ num_cores` (each core processes thousands of
  output tiles).
- **Note from verifier**: This is unlikely to be a real issue for glu_fused
  at any practical shape — the CBs are tiny. Listed for protocol
  completeness; may be a no-op refinement.
