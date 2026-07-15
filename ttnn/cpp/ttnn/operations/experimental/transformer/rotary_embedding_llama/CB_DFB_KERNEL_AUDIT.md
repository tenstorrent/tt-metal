# CB→DFB Kernel Audit: `rotary_embedding_llama`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama/`

**Scope:** `rotary_embedding_llama` program factory (interleaved + prefill-sharded) → kernels:
- compute: `device/kernels/compute/rotary_embedding_llama.cpp`, `device/kernels/compute/rotary_embedding_llama_sharded.cpp`
- dataflow: `device/kernels/dataflow/reader_rotary_embedding_llama_interleaved_start_id.cpp`, `..._llama_prefill_sharded.cpp`, `device/kernels/dataflow/writer_rotary_embedding_llama_interleaved_start_id.cpp`

Includes are all standard API headers (`api/compute/*`, `api/dataflow/*`).

## Overall verdict: GREEN

**Summary:** Llama rotary uses a transformation-matrix matmul (`rotated = x @ trans_mat`) followed by mul cos / mul sin / add. All CBs use canonical `reserve_back`/`push_back`/`wait_front`/`pop_front` on the `CircularBuffer` object API; matmul/eltwise consume via CB ids only. No `get_local_cb_interface` field access, no `read_tile_value`/`get_tile_address`, no `get_pointer_to_cb_data`, no ptr surgery. All Class 1 → mechanical `CircularBuffer` → `DataflowBuffer` rename.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `in_cb` | 1 | reader/compute | Portable | activation input, linear FIFO | Portable | — |
| `cos_cb` / `sin_cb` | 1 | reader/compute | Portable | cos/sin operands, linear FIFO | Portable | — |
| `trans_mat_cb` | 1 | reader/compute | Portable | transformation matrix for rotate; `wait_front`/`pop_front` once | Portable | — |
| `rotated_in_interm_cb` / `cos_interm_cb` / `sin_interm_cb` | 1 | compute | Portable | matmul + mul intermediates, linear FIFO | Portable | — |
| `out_cb` | 1 | compute/writer | Portable | pack → output, ptr as L1/NoC addr | Portable | — |
| `cb_id_zero` (interleaved reader zero-pad) | 1 | `reader_rotary_embedding_llama_interleaved_start_id.cpp` | Portable | zero-fill helper CB; `reserve_back`/`get_write_ptr()`/`push_back` L1 addr only | Portable | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)
