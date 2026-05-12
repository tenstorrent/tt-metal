# Capabilities: glu_fused

> Last updated: 2026-05-12 by Refinement 2 (bfloat16 support)

| Dimension | Status | Details |
|-----------|--------|---------|
| **Data formats** | float32 and bfloat16 | Entry point raises `ValueError` for any dtype ∉ {`ttnn.float32`, `ttnn.bfloat16`}. bfloat8_b: not implemented. Compute config is dtype-aware: fp32 uses HiFi4 + fp32_dest_acc + UnpackToDestFp32 (max-precision); bf16 uses LoFi + no fp32_dest_acc + default unpack (no-overhead — running fp32 settings on bf16 inputs benchmarked ~1.3× slower with zero precision gain). |
| **Layouts** | TILE only | Entry point at `glu_fused.py:71-72` raises `ValueError` for any layout ≠ `ttnn.TILE_LAYOUT`. No in-kernel tilize; ROW_MAJOR requires host-side `.to_layout(TILE)` first. |
| **Memory configs** | DRAM and L1 interleaved | Output inherits `input_tensor.memory_config()` (`glu_fused.py:41`). Both reader and writer use `TensorAccessor` with `TensorAccessorArgs` compiled in, so both DRAM and L1 interleaved work. **Sharded**: not supported (no sharded path in the program descriptor; reader assumes interleaved). |
| **Core count** | Multi-core | `ttnn.split_work_to_cores(grid_size, total_output_tiles)` over the full `compute_with_storage_grid_size` (e.g. 8×8 on Wormhole). Two-group remainder handling. |
| **Compute config** | Hard-coded, dtype-aware, not exposed | fp32: `HiFi4 + fp32_dest_acc_en=True + UnpackToDestFp32` on input CBs. bf16: `LoFi + fp32_dest_acc_en=False`, default unpack. Set in the program descriptor; entry point does NOT accept a `compute_kernel_config` kwarg — `glu_fused(input_tensor)` is the entire signature. (Note: `HiFi4` has no effect on this kernel because the only multiply is an SFPU op, not an FPU op.) |
| **Shape support** | Tile-aligned only | `W % 64 == 0` and `H % 32 == 0` enforced at `glu_fused.py:78-85`. Non-tile-aligned shapes raise `ValueError`. No padding/masking path. |
| **Rank support** | rank == 4 only | Entry point at `glu_fused.py:75-76` rejects any rank ≠ 4. PyTorch's `glu` supports arbitrary rank with a configurable `dim` — this op locks rank 4 and `dim=-1`. |
| **`dim` parameter** | dim=-1 only (not exposed) | The split is hard-coded to the last dim. No `dim` parameter on the entry point. PyTorch supports `dim ∈ [-rank, rank-1]`. |
| **Features vs PyTorch** | Subset | `torch.nn.functional.glu(x, dim)` accepts any `dim` and any rank ≥ 1 as long as `x.shape[dim] % 2 == 0`. The fused op fixes `dim=-1`, requires rank 4, requires `shape[-1] % 64 == 0` (stricter than `% 2 == 0`), and requires `shape[-2] % 32 == 0`. |
| **Acceptance tolerances** | PCC ≥ 0.999, max_abs ≤ 0.05, rel_RMS ≤ 1e-3 | Defined in `tests/ttnn/unit_tests/operations/glu_fused/test_glu_fused.py:44-46`. Achieved precision (per Phase 0 measurement) is at the fp32 floor — see `verification_report.md`. |

## Notes for Refinement Agents

- The PyTorch parity matrix above directly maps to refinement candidates: each
  "Subset" row is a potential capability expansion.
- The kernel chain (`Load A → Load B → Sigmoid → Mul`) is the only compute
  body — adding compute knobs (e.g., exposing `math_approx_mode` to toggle
  fast-approx sigmoid) is straightforward.
- The reader's split arithmetic (`a_tile_idx`, `b_tile_idx`) is hard-coded to
  split at the last dim. Generalizing to arbitrary `dim` requires recomputing
  the offset formula at host-side and passing different CT args; the kernel
  body otherwise stays the same.
