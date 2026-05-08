# Capabilities: backward_softmax

> Last updated: 2026-05-08 by incremental-verifier (Phase 0)

| Dimension | Status | Details |
|-----------|--------|---------|
| **Data formats** | float32 only | `_validate` rejects any dtype other than `ttnn.float32` (`backward_softmax.py:68-80`). bfloat16, bfloat8_b, int32 etc. → `ValueError`. Both inputs must match. |
| **Layouts** | TILE only | `_validate` rejects `ROW_MAJOR_LAYOUT` for both inputs (`backward_softmax.py:83-90`). Caller must `to_layout(TILE_LAYOUT)` first. |
| **Memory configs** | DRAM or L1 interleaved | The entry point's `memory_config` kwarg routes the *output* (default `DRAM_MEMORY_CONFIG`); inputs may live in DRAM or L1 (the kernel uses `TensorAccessor` which handles either). No sharded path. |
| **Core count** | **Single-core** | The program descriptor pins `core_grid = CoreRange(CoreCoord(0,0))` (`backward_softmax_program_descriptor.py:81-82`). All lanes for the full tensor are processed serially on Tensix (0,0). |
| **Compute config** | Hard-coded, not exposed | Entry point does **not** accept `compute_kernel_config`. Internally pins `MathFidelity.HiFi4` + `fp32_dest_acc_en=True` (`backward_softmax_program_descriptor.py:251-258`). |
| **Shape support** | Tile-aligned only, rank exactly 4 | `_validate` requires `len(shape) == 4` and `H % 32 == 0`, `W % 32 == 0` (`backward_softmax.py:93-115`). Non-aligned or rank ≠ 4 → `ValueError`. |
| **Rank support** | 4 only | Rank 2, 3, 5+ all rejected by `_validate`. |
| **Reduction dim** | `dim ∈ {-1, -2}` | Both literally rejected outside this set (`backward_softmax.py:118-121`). Positive `dim` rejected. `dim=-3` and `dim=-4` are documented as invalid because the algorithm streams over the reduction axis and the reader formula only covers W and H. |
| **Optional inputs** | None | Required: `grad_output`, `output`. Optional: `dim` (default `-1`), `memory_config` (default DRAM). No mask, no `keepdim`, no axis-broadcast variants. |
| **Features vs PyTorch** | Limited subset | PyTorch's `softmax_backward_data` (and the equivalent `nn.functional.softmax(...).backward(...)` chain) supports any reduction dim and arbitrary input rank. TTNN: only `dim ∈ {-1, -2}`, only rank-4, only float32. The numerical formula (`y * (dy − sum(y * dy, dim))`) is identical when `output` is a forward softmax result. |
| **Block size selection** | Compile-time, deterministic | `_pick_block_size` chooses the largest divisor of the reduce-axis tile count that is ≤ 8 (`backward_softmax_program_descriptor.py:24-40`). A `block_size` override exists in `create_program_descriptor` but is **not** exposed to the caller. |
| **Numerical accuracy** | PCC ≥ 0.9999 across the verified shape set | Hardware-precision-limited. fp32 dest accumulation in the matmul-based REDUCE_ROW SUM path produces ~1e-3-relative error in the row sum; after the broadcast subtract this propagates to absolute errors of ~0.1 at positions where `dy_i ≈ s`. The spec test's `atol=0.01` fails for shapes whose reduce-axis tile count ≥ 2 — see `verification_report.md`. |
