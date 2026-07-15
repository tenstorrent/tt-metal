# CB→DFB Kernel Audit: `nlp_create_qkv_heads_falcon7b`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_falcon7b/`

**Scope:** `device/kernels/dataflow/writer_tm_tile_layout_nlp_create_qkv_heads_falcon7b.cpp`. Single writer; no shared donor kernels beyond `api/` headers.

## Overall verdict: GREEN

**Summary:** Falcon-7B fused-QKV head split writer. Drains `cb_out0` (shared id with the reader's input CB) via canonical `wait_front`/`pop_front`, using `get_read_ptr()` as the NoC write source for Q/K/V destinations. Step-4 litmus scans return **zero** hits.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_id_out0` (c_0) | 1 | `writer_tm_tile_layout_nlp_create_qkv_heads_falcon7b.cpp` | Portable | canonical `wait_front`/`pop_front`, `get_read_ptr()` as NoC source → `DataflowBuffer` | Portable | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)
