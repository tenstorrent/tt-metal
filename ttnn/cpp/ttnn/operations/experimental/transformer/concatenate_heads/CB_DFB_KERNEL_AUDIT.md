# CB→DFB Kernel Audit: `concatenate_heads`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/transformer/concatenate_heads/`

**Scope:** `device/kernels/dataflow/reader_tm_tile_layout_concat_heads.cpp`, `device/kernels/dataflow/writer_tm_tile_layout_concat_heads.cpp`. No shared donor kernels beyond `api/` headers.

## Overall verdict: GREEN

**Summary:** Pure dataflow reshuffle. Reader fills `cb_in0` via `TensorAccessor` + `reserve_back`/`push_back`; writer drains `cb_out0` via `wait_front`/`pop_front`. Bare `get_read_ptr()`/`get_write_ptr()` are L1/NoC byte addresses only. Step-4 litmus scans return **zero** hits.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_id_in0` | 1 | `reader_tm_tile_layout_concat_heads.cpp` | Portable | canonical linear FIFO → `DataflowBuffer` | Portable | — |
| `cb_id_out0` | 1 | `writer_tm_tile_layout_concat_heads.cpp` | Portable | `wait_front`/`pop_front`, `get_read_ptr()` as NoC source | Portable | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)
