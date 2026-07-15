# CB→DFB Kernel Audit: `nlp_concat_heads`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads/`

**Scope:** `device/kernels/dataflow/reader_tm_tile_layout_nlp_concat_heads.cpp` (interleaved), `device/kernels/dataflow/reader_tm_tile_layout_nlp_concat_heads_sharded.cpp` (sharded). No shared donor kernels beyond `api/` headers.

## Overall verdict: GREEN

**Summary:** Dataflow reshuffle concatenating attention heads. Interleaved path uses canonical `cb_in0` FIFO (`reserve_back`/`push_back`); sharded path moves `cb_in0` → `cb_out0` in place via `reserve_back` + `get_write_ptr()` byte copies. Step-4 litmus scans return **zero** hits.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_id_in0` | 1 | `reader_*_concat_heads.cpp`, `reader_*_concat_heads_sharded.cpp` | Portable | canonical linear FIFO / input shard `get_read_ptr()` source → `DataflowBuffer` | Portable | — |
| `cb_id_out0` | 1 | `reader_*_concat_heads_sharded.cpp` | Portable | output shard, `reserve_back` + `get_write_ptr()` L1 byte copies | Portable | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)
