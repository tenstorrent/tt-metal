# CB→DFB Kernel Audit: `nlp_kv_cache_load_slice`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_kv_cache_load_slice/`

**Scope:** `nlp_kv_cache_load_slice` program factory → kernels: `device/kernels/dataflow/reader_unary_unpad_dims_interleaved_start_id_shard_optimized.cpp`. Reader-only op (no writer/compute kernel; output is a resident shard). Includes are all standard API headers.

## Overall verdict: GREEN

**Summary:** Single reader kernel that unpads/slices an interleaved KV tensor into one sharded output CB. Uses `reserve_back(num_tiles)` once, walks `get_write_ptr()` + `tile_size` offsets across `noc.async_read`s, then a single `push_back(num_tiles)`. Bare `get_write_ptr()` is an L1 byte cursor only — no `LocalCBInterface` field access. Class 1 linear FIFO (bulk-reserve variant). Zero GATE / silent-wrong / ptr-surgery / runtime-blocker hits. Mechanical `CircularBuffer` → `DataflowBuffer` rename.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_in0` | 1 | `reader_unary_unpad_dims_interleaved_start_id_shard_optimized.cpp` | Portable | bulk `reserve_back(num_tiles)` + per-tile `get_write_ptr()+offset` NoC read + single `push_back`; ptr used only as L1 addr | Portable | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)
