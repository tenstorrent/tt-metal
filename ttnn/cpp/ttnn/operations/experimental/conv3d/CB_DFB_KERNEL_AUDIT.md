# CB→DFB Kernel Audit: `conv3d`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/conv3d/`

**Scope:** `conv3d_program_factory` → kernels: `device/kernels/reader_vol2col.cpp`, `device/kernels/compute.cpp`, `device/kernels/writer.cpp`. Include closure: op-local `device/kernels/conv3d_gather_tuning.hpp`, `device/kernels/conv3d_weight_share.hpp`; `kernel_lib/tilize_helpers.hpp`, `kernel_lib/untilize_helpers.hpp`, `kernel_lib/dest_helpers.hpp`.

## Overall verdict: GREEN

**Summary:** 3D convolution via vol2col + tiled matmul (reader vol2col → tilize/matmul/reduce compute → writer). Litmus scans over all kernels + op-local/`kernel_lib` headers find **zero** `get_local_cb_interface`/`cb_interface.` access, **zero** `get_cb_tiles_*_ptr`, **zero** `read_tile_value`/`get_tile_address`, **zero** `get_pointer_to_cb_data`, and **zero** `fifo_*` pointer surgery or field reads. All CBs are canonical Class 1 linear FIFO; cross-core coordination uses `CoreLocalMem`/`noc_semaphore` (sanctioned APIs), not `LocalCBInterface` field access. Mechanical `CircularBuffer` → `DataflowBuffer` rename only.

## CB portability

CBs collapsed by role (`_cb`/`_obj`/`_id` aliases are the same buffer's handle / CT-arg id).

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_input_shard`, `cb_vol2col_rm`, `cb_vol2col_tiled` | 1 | `reader_vol2col.cpp`, `compute.cpp` | Portable | vol2col activation inputs + tilize, linear FIFO → `DataflowBuffer` | Portable | — |
| `cb_weight_tiled`, `cb_bias_tiled` | 1 | `compute.cpp`, `writer.cpp` | Portable | weight/bias inputs, linear FIFO | Portable | — |
| `cb_matmul_interm_tiled`, `cb_reduction_tiled`, `cb_interm` | 1 | `compute.cpp` | Portable | matmul/reduction intermediates, canonical `reserve/push` ↔ `wait/pop` (no rd/wr ptr surgery) | Portable | — |
| `cb_matmul_result_rm`, `cb_out` | 1 | `compute.cpp`, `writer.cpp` | Portable | pack → output, `get_read_ptr()`/`get_write_ptr()` as L1/NoC addr only | Portable | — |
| `cb_ack`, `cb_worker_ack_back`, `cb_read_offset`, `cb_dram_read_scratch` | 1/6 | `writer.cpp`, `reader_vol2col.cpp` | Portable | control/ack + DRAM scratch buffers; cross-core sync via `CoreLocalMem`/`noc_semaphore` (canonical) | Portable | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)
