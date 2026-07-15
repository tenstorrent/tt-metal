# CB→DFB Kernel Audit: `experimental/quasar/binary`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/quasar/binary/`

**Scope:** All device kernels under `device/kernels/`: compute (`eltwise_binary_kernel.cpp`, `eltwise_binary_sfpu_kernel.cpp`) and dataflow (`reader_binary_interleaved_start_id.cpp`).

## Overall verdict: GREEN

**Summary:** Clean. All Step-4 litmus scans (GATE `get_local_cb_interface(...).`/`cb_interface.`, silent-wrong `get_cb_tiles_*_ptr`, QUASAR-BLOCKED `read_tile_value`/`get_tile_address`, LTA `get_pointer_to_cb_data`, ptr-surgery `fifo_wr_ptr`/`fifo_rd_ptr`/`push_back_hold`/`llk_push_pages`, field-reads `fifo_page_size`/`fifo_num_pages`/`fifo_size`/`fifo_limit`) return **zero** hits. Two binary inputs and one output flow through canonical `reserve_back`/`push_back`/`wait_front`/`pop_front` FIFOs; bare `get_read_ptr()`/`get_write_ptr()` are used only as L1/NoC byte addresses. Mechanical `CircularBuffer` → `DataflowBuffer` rename on both arches.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_in0`, `cb_in1` | 1 | `reader_binary_interleaved_start_id.cpp`, `eltwise_binary_kernel.cpp`, `eltwise_binary_sfpu_kernel.cpp` | Portable | operand inputs, linear FIFO → `DataflowBuffer` | Portable | — |
| `cb_out0` | 1 | `eltwise_binary_kernel.cpp`, `eltwise_binary_sfpu_kernel.cpp` | Portable | pack → output, `get_write_ptr()` as L1/NoC addr only | Portable | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)
