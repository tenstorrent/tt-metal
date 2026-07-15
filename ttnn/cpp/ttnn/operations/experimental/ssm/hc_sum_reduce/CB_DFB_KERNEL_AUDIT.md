# CB→DFB Kernel Audit: `hc_sum_reduce`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/ssm/hc_sum_reduce/`

**Scope:** `hc_sum_reduce_program_factory` → kernels: `device/kernels/reader_ssm_1d_sum_reduce.cpp`, `device/kernels/ssm_1d_sum_reduce.cpp`, `device/kernels/writer_ssm_1d_sum_reduce.cpp`. Include closure: `kernel_lib/reduce_helpers_compute.hpp`, `kernel_lib/reduce_helpers_dataflow.hpp`.

## Overall verdict: GREEN

**Summary:** SSM 1D sum-reduce pipeline (reader → transpose/reduce compute → writer). Litmus scans find **zero** `get_local_cb_interface`/`cb_interface.` access, **zero** `get_cb_tiles_*_ptr`, **zero** `read_tile_value`/`get_tile_address`, **zero** `get_pointer_to_cb_data`, and **zero** `fifo_*` surgery or field reads. All CBs are canonical Class 1 linear FIFO. Mechanical `CircularBuffer` → `DataflowBuffer` rename only.

## CB portability

CBs collapsed by role (`_obj`/`_id*` aliases are the same buffer's handle / CT-arg id).

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_in0` | 1 | `reader_ssm_1d_sum_reduce.cpp`, `ssm_1d_sum_reduce.cpp` | Portable | input tiles, linear FIFO → `DataflowBuffer` | Portable | — |
| `cb_scalar` | 1 | `reader_ssm_1d_sum_reduce.cpp`, `ssm_1d_sum_reduce.cpp` | Portable | reduce scalar, linear FIFO | Portable | — |
| `cb_intermed1`, `cb_intermed2` | 1 | `ssm_1d_sum_reduce.cpp` | Portable | transpose/reduce intermediates, canonical `reserve/push` ↔ `wait/pop` | Portable | — |
| `cb_output` | 1 | `ssm_1d_sum_reduce.cpp`, `writer_ssm_1d_sum_reduce.cpp` | Portable | pack → output, `get_read_ptr()`/`get_write_ptr()` as L1/NoC addr only | Portable | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)
