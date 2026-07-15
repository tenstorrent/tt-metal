# CB→DFB Kernel Audit: `rotary_embedding_llama_fused_qk`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama_fused_qk/`

**Scope:** `rotary_embedding_llama_fused_qk_program_factory.cpp` — **compute-only** op (Q and K share resident sharded CBs; no reader/writer kernels created, only a compute `KernelDescriptor`). Kernels (selected by `row_major_input`):
- `device/kernels/compute/rotary_embedding_llama_sharded.cpp`
- `device/kernels/compute/rotary_embedding_llama_sharded_row_major.cpp`

Includes are all standard API headers (`api/compute/*`, `api/dataflow/circular_buffer.h`).

## Overall verdict: GREEN

**Summary:** Fused Q+K llama rotary that transforms both Q and K in place on sharded CBs (trans-matrix matmul + mul cos/sin + add). All CBs use canonical `reserve_back`/`push_back`/`wait_front`/`pop_front` on the `CircularBuffer` object API. No `get_local_cb_interface` field access, no `read_tile_value`/`get_tile_address`, no `get_pointer_to_cb_data`, no ptr surgery. All Class 1 → mechanical `CircularBuffer` → `DataflowBuffer` rename.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `q_in_cb` / `q_out_cb` | 1 | `..._sharded_row_major.cpp` | Portable | Q input/output sharded CBs, linear FIFO | Portable | — |
| `k_in_cb` / `k_out_cb` | 1 | `..._sharded_row_major.cpp` | Portable | K input/output sharded CBs, linear FIFO | Portable | — |
| `in_cb` / `out_cb` | 1 | `..._sharded.cpp` | Portable | tile-layout variant input/output, linear FIFO | Portable | — |
| `cos_cb` / `sin_cb` | 1 | both compute kernels | Portable | cos/sin operands, linear FIFO | Portable | — |
| `trans_mat_cb` | 1 | both compute kernels | Portable | transformation matrix; `wait_front`/`pop_front` | Portable | — |
| `rotated_in_interm_cb` / `cos_interm_cb` / `sin_interm_cb` | 1 | both compute kernels | Portable | matmul + mul intermediates, linear FIFO | Portable | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)
