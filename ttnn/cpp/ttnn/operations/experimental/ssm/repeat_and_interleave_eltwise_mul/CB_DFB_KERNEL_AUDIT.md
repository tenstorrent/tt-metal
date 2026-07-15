# CB→DFB Kernel Audit: `repeat_and_interleave_eltwise_mul`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/ssm/repeat_and_interleave_eltwise_mul/`

**Scope:** program factory → kernels: `device/kernels/reader_ssm_eltwise_mul.cpp`, `device/kernels/ssm_eltwise_mul.cpp`, `device/kernels/writer_ssm_eltwise_mul.cpp`.

## Overall verdict: GREEN

**Summary:** SSM repeat/interleave + elementwise-multiply pipeline (reader → transpose/bcast/mul compute → writer). Litmus scans find **zero** `get_local_cb_interface`/`cb_interface.` access, **zero** `get_cb_tiles_*_ptr`, **zero** `read_tile_value`/`get_tile_address`, **zero** `get_pointer_to_cb_data`, and **zero** `fifo_*` surgery or field reads. All CBs are canonical Class 1 linear FIFO. The `*_read_ptr`/`*_write_ptr` local variables are values returned by `get_read_ptr()`/`get_write_ptr()` (bare L1 addresses), not `fifo_rd_ptr`/`fifo_wr_ptr` field access. Mechanical `CircularBuffer` → `DataflowBuffer` rename only.

## CB portability

CBs collapsed by role (`_buf` aliases are the same buffer's handle).

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_in0`, `cb_in1` | 1 | `reader_ssm_eltwise_mul.cpp`, `ssm_eltwise_mul.cpp` | Portable | inputs, linear FIFO → `DataflowBuffer` | Portable | — |
| `cb_in0_transposed`, `cb_in1_transposed`, `cb_in1_bcast_row` | 1 | `ssm_eltwise_mul.cpp` | Portable | transpose/broadcast intermediates, canonical `reserve/push` ↔ `wait/pop` (`get_read_ptr()`/`get_write_ptr()` as L1 addr) | Portable | — |
| `cb_out`, `cb_out_transposed` | 1 | `ssm_eltwise_mul.cpp`, `writer_ssm_eltwise_mul.cpp` | Portable | pack → output, L1/NoC addr only | Portable | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)
