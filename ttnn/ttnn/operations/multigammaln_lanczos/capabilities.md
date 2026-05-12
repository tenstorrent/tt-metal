# Capabilities: multigammaln_lanczos

> Last updated: 2026-05-12 by ttnn-verifier (Phase 0)

Source of truth for what `multigammaln_lanczos` currently accepts. Derived from the entry point validation (`multigammaln_lanczos.py:54-78`), the program descriptor (`multigammaln_lanczos_program_descriptor.py`), and the three kernels under `kernels/`. Update this table when a refinement broadens the supported surface.

| Dimension | Status | Details |
|-----------|--------|---------|
| **Data formats** | float32 only | Entry point validates `input_tensor.dtype == ttnn.float32` (`multigammaln_lanczos.py:62`). bfloat16/bfloat8_b inputs raise `ValueError`. |
| **Layouts** | TILE only | Entry point validates `input_tensor.layout == ttnn.TILE_LAYOUT` (`multigammaln_lanczos.py:67`). ROW_MAJOR inputs raise `ValueError`; no in-kernel tilize/untilize path. |
| **Memory configs** | DRAM **and** L1 interleaved | Output inherits `input_tensor.memory_config()` (`multigammaln_lanczos.py:35`). Kernels read/write via `TensorAccessor`, which transparently handles both DRAM and L1 interleaved layouts. Sharded layouts: not supported — `TensorAccessorArgs(tensor).get_compile_time_args()` would not include shard CT args because no sharded shard spec is plumbed; `cb_input_tiles` / `cb_output_tiles` are not sharded CBs. Verified on `L1_MEMORY_CONFIG` by `test_l1_interleaved_input` in `test_multigammaln_lanczos_extended.py`. |
| **Core count** | Multi-core | Program descriptor calls `ttnn.split_work_to_cores(grid_size, total_tiles)` and assigns per-core RT args via the standard group_1 + group_2 walk (`multigammaln_lanczos_program_descriptor.py:48-133`). All grid cores participate when `total_tiles ≥ num_cores`. |
| **Compute config** | **Hard-coded** | The Python entry point does not accept `compute_kernel_config`. The program descriptor pins `math_fidelity=HiFi4`, `fp32_dest_acc_en=True`, and (as of Phase 0 verification) `unpack_to_dest_mode[cb_input_tiles] = unpack_to_dest_mode[cb_accumulator] = UnpackToDestFp32`. Caller cannot override. |
| **Shape support** | Tile-aligned only, rank ≥ 2 | `H % 32 == 0` and `W % 32 == 0` enforced at the entry point (`multigammaln_lanczos.py:74`). Rank < 2 rejected (`multigammaln_lanczos.py:71`). No padding path inside the kernel. |
| **Rank support** | 2D, 3D, 4D, … | Validator requires `len(shape) >= 2` but imposes no upper bound. Internally the kernel works on a flat tile-id stream — rank is irrelevant past the validator. Tested on rank 4 only (acceptance test) but should work for any rank ≥ 2. |
| **`p` parameter** | Fixed at 4 | Permanently hard-coded — matches `torch.special.multigammaln(x, p=4)`. Not a refinement target (see `op_design.md` § Out of Scope). |
| **Output `memory_config` kwarg** | Not exposed | Output inherits the input's memory config. No way to redirect. |
| **Value domain** | `a ∈ [2.0, 10.0]` safe | Mathematically defined for `a > 1.5`; the Lanczos 6-term polynomial is stable at fp32 only on the documented safe domain. Outside this domain, inputs propagate as NaN/Inf naturally — there is no domain check at validation. Pole-zeroing masks fire correctly at exact `a = 1.0` and `a = 2.0`. |
| **Features vs PyTorch** | Subset | `torch.special.multigammaln(x, p)` allows any positive integer `p`. TTNN supports **only `p = 4`** (no `p` argument). Both implementations support arbitrary tensor shape; the TTNN side requires tile-aligned H/W. |

## Validation Surface (entry point rejects)

| Reject Reason | Where checked |
|---------------|--------------|
| `dtype != ttnn.float32` | `multigammaln_lanczos.py:62` |
| `layout != ttnn.TILE_LAYOUT` | `multigammaln_lanczos.py:67` |
| `len(shape) < 2` | `multigammaln_lanczos.py:71` |
| `shape[-1] % 32 != 0` or `shape[-2] % 32 != 0` | `multigammaln_lanczos.py:74` |
| Not on device | `multigammaln_lanczos.py:56` |
