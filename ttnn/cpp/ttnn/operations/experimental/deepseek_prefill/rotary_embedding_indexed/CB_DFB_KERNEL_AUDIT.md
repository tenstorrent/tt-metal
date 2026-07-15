# CB→DFB Kernel Audit: `rotary_embedding_indexed`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/rotary_embedding_indexed/`

**Scope:** In-scope device kernels under `device/kernels/` → `dataflow/reader_rotary_embedding_indexed_interleaved_start_id.cpp` (the only kernel in this op tree; the compute/writer are donor kernels from the base rotary-embedding op, not under this op root).

## Overall verdict: GREEN

**Summary:** The one in-scope kernel is a canonical NoC reader. All four CBs (`input`, `cos`, `sin`, `trans_mat`) are Class 1 linear FIFOs — `reserve_back` → `get_write_ptr()` (used only as an L1 write address) → `async_read` → `push_back`. No GATE, no silent-wrong, no ptr surgery, no field reads, no runtime blockers. Mechanical `CircularBuffer` → `DataflowBuffer` rename.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `input_cb` | 1 | `reader_rotary_embedding_indexed_interleaved_start_id.cpp` | Portable | activation input; `reserve_back`/`get_write_ptr()`/`push_back` linear FIFO | Portable | — |
| `cos_cb` | 1 | `reader_rotary_embedding_indexed_interleaved_start_id.cpp` | Portable | cos shard read, linear FIFO; `get_write_ptr()` as L1 addr only | Portable | — |
| `sin_cb` | 1 | `reader_rotary_embedding_indexed_interleaved_start_id.cpp` | Portable | sin shard read, linear FIFO | Portable | — |
| `trans_mat_cb` | 1 | `reader_rotary_embedding_indexed_interleaved_start_id.cpp` | Portable | transform matrix, read once then reused; linear FIFO | Portable | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)
