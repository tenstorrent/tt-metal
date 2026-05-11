# Capabilities: multigammaln_lanczos

> Last updated: 2026-05-11 by incremental-verifier (Phase 0)

Multivariate log-gamma at order p = 4, implemented via the **Lanczos 6-term
polynomial** as a single fused TTNN kernel (NOT the Stirling+reflection variant
used by `ttnn.multigammaln`):

`multigammaln_lanczos(a) = L(a) + L(a-0.5) + L(a-1.0) + L(a-1.5) + 3·log(π)`

where `L(.)` is the Lanczos polynomial approximation of `lgamma`. The order is
hard-coded; the entry point does not accept a `p` argument and the compute
kernel deliberately does NOT use any SFPU `lgamma_*` helper.

| Dimension | Status | Details |
|-----------|--------|---------|
| **Data formats** | `float32` only | Entry point validates `input_tensor.dtype == ttnn.float32` and raises `ValueError` otherwise (`multigammaln_lanczos.py:65`). `bfloat16`, `bfloat8_b`, integer dtypes: not handled. Hard-coded `fp32_dest_acc_en=True` in the program descriptor reflects this. |
| **Layouts** | `TILE_LAYOUT` only | Entry point validates layout and rejects `ROW_MAJOR_LAYOUT` (`multigammaln_lanczos.py:70`). No in-kernel tilize/untilize path. Row-major inputs must be staged through host `.to_layout(TILE)`. |
| **Memory configs** | DRAM or L1 interleaved (input) / either (output) | Reader uses `TensorAccessor` on the interleaved layout; output `memory_config` is plumbed through to `ttnn.allocate_tensor_on_device` (defaults to `DRAM_MEMORY_CONFIG`). Verified for L1 output by `test_multigammaln_lanczos_extended.py::test_multigammaln_lanczos_output_memory_config[l1_output]`. No sharded path. |
| **Core count** | Multi-core | `ttnn.split_work_to_cores(grid_size, total_tiles)` in `multigammaln_lanczos_program_descriptor.py:55–62`. Reader/writer/compute all placed on `all_cores`. |
| **Compute config** | Hard-coded internally | Program descriptor pins `math_fidelity=HiFi4`, `fp32_dest_acc_en=True`, and sets `unpack_to_dest_mode=UnpackToDestFp32` on every fp32 CB. Entry point does **not** accept a `compute_kernel_config=` kwarg. |
| **Shape support** | Rank-4 `(N, C, H, W)`, `H % 32 == 0`, `W % 32 == 0` | Entry point validates (`multigammaln_lanczos.py:76–84`). Non-aligned dims raise `ValueError`. No padding/masking path. |
| **Rank support** | Rank 4 only | Rank ≠ 4 → `ValueError`. No internal reshape to 4D. |
| **Features vs PyTorch** | Order `p = 4` only | `torch.special.multigammaln(x, p)` supports any `p >= 1`. This op pins `p = 4` (four lgamma terms + the `(p*(p-1)/4)*log(π) = 3*log(π)` constant baked into the kernel). |
| **Domain handling** | `a > 1.5` is the strict mathematical domain | No input branching. The Lanczos polynomial naturally produces NaN/-Inf for inputs with `a ≤ 1.5` (negative `input+j` makes the series go negative → log(neg) = NaN). **Diverges from `torch.special.multigammaln` on OOD inputs**: torch returns finite values for `a ∈ (0, 1.5]` (it just sums lgammas, which are finite for negative non-integer args), but the Lanczos kernel returns NaN. This is intrinsic to the Lanczos approximation and matches the design contract. |
| **Pole zeroing** | At exact integer poles `a == offset + 1` and `a == offset + 2` for each offset in `{0, 0.5, 1.0, 1.5}` | Implemented via `unary_eq_tile` → `rsub_unary_tile` → `mul_binary_tile` (no input-value branch). Only fires for bitwise-exact poles (probability 0 for random fp32). |
| **Precision** | Phase-0 baseline | PCC ≈ 0.99999999, max abs ≈ 5.2e-3, mean abs ≈ 8e-4, relative RMS ≈ 5.5e-5 across the in-domain test shapes (see `verification_report.md` and `test_multigammaln_lanczos_precision_baseline.py`). With `UnpackToDestFp32` on every fp32 CB, the tile-boundary round-trip is bit-exact end-to-end. |
