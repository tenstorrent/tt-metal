# CB→DFB Kernel Audit: `split_query_key_value_and_split_heads`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/transformer/split_query_key_value_and_split_heads/`

**Scope:** `split_query_key_value_and_split_heads` program factory (interleaved + sharded) → kernels:
- compute: `device/kernels/compute/transpose_wh_sharded.cpp`
- dataflow: `device/kernels/dataflow/reader_tm_tile_layout_create_qkv_heads.cpp`, `..._create_qkv_heads_sharded.cpp`, `device/kernels/dataflow/writer_tm_tile_layout_create_qkv_heads.cpp`, `..._create_qkv_heads_sharded.cpp`

Includes are all standard API headers (`api/dataflow/*`, `api/dataflow/endpoints.h`, `api/compute/transpose.h`).

## Overall verdict: GREEN

**Summary:** QKV split + head reshape with an optional `transpose_wh` on K heads. Reader fills input CBs, compute transposes tile-by-tile (`wait_front`/`transpose_tile`/`pack_tile`/`push_back`), writer drains Q/K/V to their output tensors. All CBs use canonical `reserve_back`/`push_back`/`wait_front`/`pop_front` on the `CircularBuffer` object API; `get_read_ptr()`/`get_write_ptr()` used only as L1/NoC addresses. No `get_local_cb_interface` field access, no ptr surgery, no runtime-blocked APIs. All Class 1 → mechanical `CircularBuffer` → `DataflowBuffer` rename.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_in0` (c_0) | 1 | reader/writer | Portable | Q/V head tiles, linear FIFO; ptr as L1/NoC addr | Portable | — |
| `cb_in1` (c_1) | 1 | reader | Portable | K head tiles feeding transpose, linear FIFO | Portable | — |
| `cb_im0` (c_24) | 1 | `transpose_wh_sharded.cpp` | Portable | transpose input tile, `wait_front`/`pop_front` | Portable | — |
| `cb_out0` / `cb_out1` (c_16 / c_17) | 1 | `transpose_wh_sharded.cpp`, writer | Portable | transposed K output; pack → `push_back` → NoC write | Portable | — |
| `cb_id_out*` (writer output CBs, incl. c_18) | 1 | writer(s) | Portable | Q/K/V outputs, linear FIFO; `get_read_ptr()` as NoC addr | Portable | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)
