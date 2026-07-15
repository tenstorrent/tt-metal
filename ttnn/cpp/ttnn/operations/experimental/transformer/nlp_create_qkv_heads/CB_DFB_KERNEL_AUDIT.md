# CB→DFB Kernel Audit: `nlp_create_qkv_heads`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/`

**Scope:** `device/kernels/dataflow/reader_tm_tile_layout_nlp_create_qkv_heads.cpp` (interleaved), `device/kernels/dataflow/reader_tm_tile_layout_nlp_create_qkv_heads_sharded.cpp` (sharded), `device/kernels/dataflow/writer_tm_tile_layout_nlp_create_qkv_heads.cpp`. No shared donor kernels beyond `api/` headers.

## Overall verdict: GREEN

**Summary:** QKV head-split dataflow reshuffle. `cb_qv`/`cb_k` (interleaved) and `cb_q_out`/`cb_kv_out` (sharded) are canonical Class 1 FIFOs driven by `CircularBuffer` `reserve_back`/`push_back`/`wait_front`/`pop_front`; the writer drains them to output. `cb_k` maps to c_16 when compute transposes K, else c_1 (direct). Step-4 litmus scans return **zero** hits.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_qv` (c_1) | 1 | `reader_*_nlp_create_qkv_heads.cpp`, `writer_*` | Portable | Q/V head FIFO → `DataflowBuffer` | Portable | — |
| `cb_k` (c_0 compute / c_1 or c_16 direct) | 1 | `reader_*_nlp_create_qkv_heads.cpp`, `writer_*` | Portable | K head FIFO; transpose variant uses c_16 (compute-filled) | Portable | — |
| `cb_q_out` | 1 | `reader_*_nlp_create_qkv_heads_sharded.cpp` | Portable | sharded Q output, canonical reserve/push + `get_write_ptr()` | Portable | — |
| `cb_kv_out` | 1 | `reader_*_nlp_create_qkv_heads_sharded.cpp` | Portable | sharded KV output, canonical reserve/push | Portable | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)
