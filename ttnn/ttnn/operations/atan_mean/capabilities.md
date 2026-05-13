# Capabilities: atan_mean

> Last updated: 2026-05-13 by ttnn-verifier (Phase 0)

| Dimension | Status | Details |
|-----------|--------|---------|
| **Data formats** | float32 only | Entry point validates `dtype == ttnn.float32`; bfloat16 / bfloat8_b raise `ValueError`. Internal scaler CB is bf16 (matmul col-0 convention), but this is invisible to the caller. |
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

Measured on shapes `[(1,1,32,32), (1,1,64,64), (1,1,256,128), (1,8,128,128)]` with N(0, 1) inputs:
- PCC ≥ 0.99999995
- Max abs error: ~1.3e-4
- Mean abs error: ~3e-5
- Relative RMS error: ~6.5e-4

Errors are dominated by SFPU atan polynomial approximation; bf16 quantisation of the `1/W` scaler is bit-exact for power-of-two W (32, 64, 128, …) so it contributes nothing for the tested shapes.

## Known limitations (Phase 0)

- **No FLOAT32 alternative**: refusal of bfloat16 is by design — refinement would add bfloat16 + bfloat8_b support.
- **No ROW_MAJOR input**: a caller passing a row-major tensor must convert first.
- **No non-tile-aligned shapes**: the partial-scaler infrastructure in the kernel-lib is available but unwired.
- **No `compute_kernel_config`**: callers cannot trade accuracy for throughput (e.g. HiFi3 / no fp32 dest accum).
- **Large `W` will L1-overflow `cb_atan_tiles`**: sized to `Wt` fp32 pages. For W=8192 → Wt=256 → 1 MB just for this CB. The current shape set caps Wt at 4 so this is not yet a problem, but it caps the supported W ceiling.
