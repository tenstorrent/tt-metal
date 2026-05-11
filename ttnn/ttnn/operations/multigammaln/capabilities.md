# Capabilities: multigammaln

> Last updated: 2026-05-11 by incremental-verifier (Phase 0)

Multivariate log-gamma at order p = 4:
`multigammaln(a) = lgamma(a) + lgamma(a-0.5) + lgamma(a-1.0) + lgamma(a-1.5) + 3*log(π)`.

Matches `torch.special.multigammaln(x, p=4)`. The order is hard-coded; the entry
point does not accept a `p` argument.

| Dimension | Status | Details |
|-----------|--------|---------|
| **Data formats** | `float32` only | Entry point validates `input_tensor.dtype == ttnn.float32` and raises `ValueError` otherwise (`multigammaln.py:59`). `bfloat16`, `bfloat8_b`, integer dtypes: not handled. Hard-coded `fp32_dest_acc_en=True` in the program descriptor reflects this. |
| **Layouts** | `TILE_LAYOUT` only | Entry point validates layout and rejects `ROW_MAJOR_LAYOUT` (`multigammaln.py:64`). No in-kernel tilize/untilize path. Row-major inputs must be staged through host `.to_layout(TILE)`. |
| **Memory configs** | DRAM or L1 interleaved (input) / either (output) | Reader uses `TensorAccessor` on the interleaved layout; output `memory_config` is plumbed through to `ttnn.allocate_tensor_on_device` (defaults to `DRAM_MEMORY_CONFIG`). No sharded path. |
| **Core count** | Multi-core | `ttnn.split_work_to_cores(grid_size, total_tiles)` in `multigammaln_program_descriptor.py:54–61`. Reader/writer/compute all placed on `all_cores`. |
| **Compute config** | Hard-coded internally | Program descriptor pins `math_fidelity=HiFi4`, `fp32_dest_acc_en=True`, and sets `unpack_to_dest_mode=UnpackToDestFp32` on every fp32 CB. Entry point does **not** accept a `compute_kernel_config=` kwarg. |
| **Shape support** | Rank-4 `(N, C, H, W)`, `H % 32 == 0`, `W % 32 == 0` | Entry point validates (`multigammaln.py:70–78`). Non-aligned dims raise `ValueError`. No padding/masking path. |
| **Rank support** | Rank 4 only | Rank ≠ 4 → `ValueError`. No internal reshape to 4D. |
| **Features vs PyTorch** | Order p = 4 only | `torch.special.multigammaln(x, p)` supports any `p >= 1`. This op pins `p = 4` (four `lgamma` terms + the `(p*(p-1)/4)*log(π) = 3*log(π)` constant baked into the kernel). |
| **Domain handling** | `a > 1.5` is the strict mathematical domain; `a ≤ 1.5` falls through to NaN/inf naturally | No input branching. Out-of-domain values propagate NaN/inf in the same positions as `torch.special.multigammaln`. With `UnpackToDestFp32` enabled on every fp32 CB, inputs within ε of an integer boundary no longer overflow to +inf in sub-phase B (a previously known limitation). |
| **Precision** | Phase-0 baseline | PCC > 0.99999, max abs ≈ 0.05, rel RMS ≈ 6e-4 across the in-domain test shapes (see `verification_report.md` and `test_multigammaln_precision_baseline.py`). |
