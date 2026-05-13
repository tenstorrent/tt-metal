# Capabilities: atan_mean

> Last updated: 2026-05-13 by ttnn-implementer (Refinement 2)

| Dimension | Status | Details |
|-----------|--------|---------|
| **Data formats** | float32, bfloat16, bfloat8_b | Entry point validates `dtype ∈ {float32, bfloat16, bfloat8_b}`; other dtypes raise `ValueError`. Output dtype matches the input dtype. Internal scaler CB is bf16 (matmul col-0 convention) regardless of input dtype. Accumulation runs in fp32 (`fp32_dest_acc_en=True`), so the dominant numerical error for bf16/bf8 inputs comes from input quantisation, not the reduce path. |
| **Layouts** | TILE only | Entry point validates `layout == ttnn.TILE_LAYOUT`. ROW_MAJOR raises `ValueError` — no in-kernel tilize path. Caller must `.to_layout(TILE_LAYOUT)` host-side first. |
| **Memory configs** | DRAM interleaved (input + output) | Output is always allocated with `ttnn.DRAM_MEMORY_CONFIG` regardless of input memory config. No sharded path. Input may be L1 interleaved or DRAM (reader uses `TensorAccessor`), but this is not exercised in tests. |
| **Core count** | Multi-core | Program descriptor uses `ttnn.split_work_to_cores(grid_size, total_row_tiles)` against the full `compute_with_storage_grid_size()`. Two-group split balances load with ≤1 row-tile difference per core. |
| **Compute config** | Not exposed | Entry point does **not** accept `compute_kernel_config`. Hard-coded: `math_fidelity=HiFi4`, `fp32_dest_acc_en=True`. `math_approx_mode` and `packer_l1_acc` are not set (descriptor defaults). |
| **Shape support** | Tile-aligned only | Validates `H % 32 == 0` and `W % 32 == 0`. Non-aligned shapes raise `ValueError`. No partial-scaler / masking path is wired up. |
| **Rank support** | Rank 4 only | Validates `len(shape) == 4`. Rank ≠ 4 raises `ValueError`. The op returns a rank-3 view via `ttnn.squeeze(out, dim=-1)` (metadata-only). |
| **Reduction dim** | `dim=-1` (W) only | Hard-coded — no `dim` or `keepdim` parameter. Reduces W axis exclusively via matmul-mode REDUCE_ROW + AVG. |
| **Features vs PyTorch** | Partial | PyTorch `torch.atan(x).mean(dim=d, keepdim=k)` supports `dim ∈ {-1, -2, -3, -4}`, `keepdim ∈ {False, True}`, and integer tensor dim lists. TTNN exposes only the fused fixed-dim=-1, keepdim=False variant. |
| **Optional inputs** | None | No optional parameters (no `dim`, no `keepdim`, no `memory_config`, no `compute_kernel_config`). |
| **Function signature** | `atan_mean(input_tensor: ttnn.Tensor) -> ttnn.Tensor` | Positional or keyword (`input_tensor=`). Returns rank-3 `(N, C, H)` float32 TILE_LAYOUT in DRAM. |

## Numerical precision

Measured on shapes `[(1,1,32,32), (1,1,64,64), (1,1,128,64), (1,1,32,96), (1,8,64,64)]` with N(0, 1) inputs:

| dtype | PCC achieved | Max abs error | Notes |
|-------|--------------|---------------|-------|
| float32   | ≥ 0.99999995 | ≈ 1.3e-4 | Unchanged from Phase 0; dominated by SFPU atan polynomial. |
| bfloat16  | ≥ 0.9998     | ≈ 6e-3   | Input bf16 quantisation (7-bit mantissa) is the dominant error; reduce accumulator is still fp32. |
| bfloat8_b | ≥ 0.997      | ≈ 4e-2   | Block-format input quantisation dominates. Output is also bf8 (matches input dtype), so an additional round-trip pack-error appears on the host. |

The `1/W` reduce scaler is bf16 and bit-exact for power-of-two W (32, 64, 128, …) — it contributes nothing for the tested shapes. The matmul-mode REDUCE_ROW path unpacks the input on SrcA into a fp32 destination accumulator, so dtype-induced error scales with the input value range, not with W.

## Known limitations

- **No ROW_MAJOR input**: a caller passing a row-major tensor must convert first.
- **No non-tile-aligned shapes**: the partial-scaler infrastructure in the kernel-lib is available but unwired.
- **No `compute_kernel_config`**: callers cannot trade accuracy for throughput (e.g. HiFi3 / no fp32 dest accum).
- **Large `W` will L1-overflow `cb_atan_tiles`**: sized to `Wt` input-dtype pages. For W=8192 with fp32 input → Wt=256 → 1 MB just for this CB. With bf16/bf8 input, the page size shrinks proportionally, so the W ceiling rises (~16K for bf16, ~32K for bf8), but the streaming-reduce refinement is still required for arbitrary W.
