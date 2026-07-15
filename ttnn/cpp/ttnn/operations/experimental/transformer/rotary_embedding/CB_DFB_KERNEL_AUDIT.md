# CB→DFB Kernel Audit: `rotary_embedding`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding/`

**Scope:** `rotary_embedding` program factory (interleaved + sharded, single-tile + multi-tile, prefill + `DECODE_MODE`) → kernels:
- compute: `device/kernels/compute/rotary_embedding.cpp`, `device/kernels/compute/rotary_embedding_single_tile.cpp`
- dataflow: `device/kernels/dataflow/reader_rotary_embedding_interleaved_start_id.cpp`, `..._interleaved_start_id_sharded.cpp`, `reader_rotary_embedding_single_tile_interleaved_start_id.cpp`, `..._single_tile_interleaved_start_id_sharded.cpp`, `device/kernels/dataflow/writer_rotary_embedding_interleaved_start_id.cpp`
- shared headers: `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp`, `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp` (both scanned — clean)

## Overall verdict: GREEN

**Summary:** Dataflow + eltwise (mul/add bcast, tilize/untilize in `DECODE_MODE`) rotary op. Every CB uses canonical `reserve_back`/`push_back`/`wait_front`/`pop_front` on the `CircularBuffer` object API; `get_write_ptr()` in the reader is used only to seed the scalar tile as an L1 address. No `get_local_cb_interface` field access, no `read_tile_value`/`get_tile_address`, no `get_pointer_to_cb_data`, no ptr surgery. All Class 1 → mechanical `CircularBuffer` → `DataflowBuffer` rename. The `DECODE_MODE` tilize/untilize helpers use canonical FIFO sync CBs (no field reads).

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_in` / `cb_rotated_in` | 1 | reader/compute | Portable | input + rotated-input activation, linear FIFO | Portable | — |
| `cb_cos` / `cb_sin` | 1 | reader/compute | Portable | cos/sin operands, linear FIFO (bulk-reserve in `DECODE_MODE`) | Portable | — |
| `cb_scalar` | 1 | reader/compute | Portable | scalar (-1) tile; reader seeds via `get_write_ptr()` L1 addr then `push_back` | Portable | — |
| `cb_rotated_in_interm` / `cb_cos_interm` / `cb_sin_interm` | 1 | `rotary_embedding.cpp` | Portable | compute intermediates, linear FIFO | Portable | — |
| `cb_out` | 1 | compute/writer | Portable | pack → output, `get_read_ptr()`/`get_write_ptr()` as L1/NoC addr | Portable | — |
| `untilized_{cos,sin}_cb`, `retilized_{cos,sin}_cb`, `untilized_{cos,sin}_sync_cb` | 1 | `rotary_embedding.cpp` (`DECODE_MODE`) | Portable | tilize/untilize staging + sync CBs, canonical `wait_front`/`pop_front` | Portable | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)
