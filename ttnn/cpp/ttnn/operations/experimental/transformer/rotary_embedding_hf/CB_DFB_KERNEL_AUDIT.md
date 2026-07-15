# CB→DFB Kernel Audit: `rotary_embedding_hf`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_hf/`

**Scope:** `rotary_embedding_hf` program factory (interleaved + sharded, single-tile variants) → kernels:
- compute: `device/kernels/compute/rotary_embedding_hf.cpp`, `..._hf_sharded.cpp`, `..._hf_single_tile_sharded.cpp`
- dataflow: `device/kernels/dataflow/reader_rotary_embedding_hf_interleaved.cpp`, `..._hf_sharded.cpp`, `..._hf_single_tile_interleaved_start_id.cpp`, `..._hf_single_tile_interleaved_start_id_sharded.cpp`, `..._hf_single_tile_sharded.cpp`, `device/kernels/dataflow/writer_rotary_embedding_hf_interleaved.cpp`
- shared header: `ttnn/cpp/ttnn/kernel/compute/dest_format_helpers.hpp` (scanned — clean)

## Overall verdict: GREEN

**Summary:** HF-style rotary op (rotate-half + mul cos/sin + add) built from dataflow readers/writers and eltwise-binary compute kernels. All CBs (`in_cb`, `rotated_in_cb`, `cos_cb`, `sin_cb`, `scalar_cb`, `rotated_in_interm_cb`, `cos_interm_cb`, `sin_interm_cb`, `out_cb`) use canonical `reserve_back`/`push_back`/`wait_front`/`pop_front` on the `CircularBuffer` object API. No `get_local_cb_interface` field access, no `read_tile_value`/`get_tile_address`, no `get_pointer_to_cb_data`, no ptr surgery. All Class 1 → mechanical `CircularBuffer` → `DataflowBuffer` rename.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `in_cb` / `rotated_in_cb` | 1 | reader/compute | Portable | input + rotated-input, linear FIFO | Portable | — |
| `cos_cb` / `sin_cb` | 1 | reader/compute | Portable | cos/sin operands, linear FIFO | Portable | — |
| `scalar_cb` | 1 | reader/compute | Portable | scalar (-1) tile; `get_write_ptr()` L1 addr only | Portable | — |
| `rotated_in_interm_cb` / `cos_interm_cb` / `sin_interm_cb` | 1 | compute | Portable | eltwise intermediates, linear FIFO | Portable | — |
| `out_cb` | 1 | compute/writer | Portable | pack → output, ptr as L1/NoC addr | Portable | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)
