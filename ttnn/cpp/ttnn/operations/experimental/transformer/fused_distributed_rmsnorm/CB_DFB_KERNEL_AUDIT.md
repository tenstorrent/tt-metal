# CB→DFB Kernel Audit: `fused_distributed_rmsnorm`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/transformer/fused_distributed_rmsnorm/`

**Scope:** `device/kernels/compute/rmsnorm_pre_allgather.cpp`, `device/kernels/compute/rmsnorm_post_allgather.cpp`, `device/kernels/dataflow/rms_pre_allgather_reader.cpp`, `rms_pre_allgather_writer.cpp`, `rms_post_allgather_reader.cpp`, `rms_post_allgather_writer.cpp`, `writer_unary_interleaved_start_id_blocked.cpp`. Donor includes: `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp`, `reduce_helpers_dataflow.hpp`, `ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp` (all scanned — clean).

## Overall verdict: GREEN

**Summary:** Distributed RMSNorm (pre/post all-gather) with optional weight and fused RoPE. **Verified against the known finding:** this family uses `reduce_helpers_compute.hpp` (not `kernel_util/compute/memory.h`) and does **not** call `get_pointer_to_cb_data`/`get_tile_address`. All CBs are canonical Class 1 linear FIFOs. The pre-allgather compute uses `llk_pack_reconfig_l1_acc(0/1)` with `pack_tile<true>` for on-chip L1 accumulation of `intermediate_cb` — this is the sanctioned `packer_l1_acc` pattern (no `fifo_*` save/restore, no `get_local_cb_interface`), so it is not Class 5 ptr surgery. Step-4 litmus scans return **zero** hits across the closure.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `input_cb` | 1 | `rmsnorm_pre_allgather.cpp`, `rmsnorm_post_allgather.cpp`, `rms_*_reader.cpp` | Portable | activation input, linear FIFO → `DataflowBuffer` | Portable | — |
| `reduce_scalar_cb` | 1 | `rmsnorm_pre_allgather.cpp`, `rmsnorm_post_allgather.cpp`, readers | Portable | reduce scalar FIFO | Portable | — |
| `intermediate_cb` | 1/5 | `rmsnorm_pre_allgather.cpp`, `rmsnorm_post_allgather.cpp` | Portable | E(x²) accumulate via canonical `packer_l1_acc` (`llk_pack_reconfig_l1_acc` + `pack_tile<true>`), no ptr surgery | Portable | — |
| `output_cb` | 1 | both compute + `rms_*_writer.cpp`, `writer_unary_interleaved_start_id_blocked.cpp` | Portable | pack → output, `get_write_ptr()` as NoC addr | Portable | — |
| `stats_cb` | 1 | `rmsnorm_post_allgather.cpp`, `rms_post_allgather_reader.cpp` | Portable | gathered per-device stats, canonical FIFO | Portable | — |
| `weight_cb` | 1 | `rmsnorm_post_allgather.cpp`, `rms_post_allgather_reader.cpp` | Portable | gamma weights, cumulative `wait_front`/`pop_front` | Portable | — |
| `epsilon_cb` | 1 | `rmsnorm_post_allgather.cpp`, `rms_post_allgather_reader.cpp` | Portable | broadcast epsilon via `generate_bcast_col_scalar`, linear FIFO | Portable | — |
| `reduce_result_cb` | 1 | `rmsnorm_post_allgather.cpp` | Portable | 1/sqrt(mean²+eps), in-place-capable canonical reserve/push | Portable | — |
| `transformation_mat_cb` | 1 | `rmsnorm_post_allgather.cpp`, `rms_post_allgather_reader.cpp` | Portable | RoPE transform matrix, linear FIFO | Portable | — |
| `rope_cos_cb` / `rope_sin_cb` | 1 | `rmsnorm_post_allgather.cpp`, `rms_post_allgather_reader.cpp` | Portable | RoPE cos/sin FIFOs | Portable | — |
| `rotated_input_cb` | 1 | `rmsnorm_post_allgather.cpp` | Portable | RoPE rotate intermediate, canonical reserve/push | Portable | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)
