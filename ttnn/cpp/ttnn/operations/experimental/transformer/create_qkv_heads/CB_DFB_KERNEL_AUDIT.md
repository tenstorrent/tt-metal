# CB→DFB Kernel Audit: `create_qkv_heads`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/transformer/create_qkv_heads/`

**Scope:** `device/kernels/reader_create_qkv_heads_sharded.cpp`. Single sharded reader; no shared donor kernels beyond `api/` headers.

## Overall verdict: GREEN

**Summary:** Sharded Q/K/V split by NoC reads from one input shard into three output shards. `cb_outq`/`cb_outk`/`cb_outv` use `reserve_back` → `get_write_ptr() + byte_offset` scatter → `push_back` (a standard sharded head-split write into a reserved region), with `cb_in0` read via `get_read_ptr()` as the NoC source. No `get_local_cb_interface`, no `fifo_*` field access, no `get_pointer_to_cb_data` — Step-4 litmus scans return **zero** hits.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_in0` | 1 | `reader_create_qkv_heads_sharded.cpp` | Portable | input shard; `get_read_ptr()` as NoC source base | Portable | — |
| `cb_outq` / `cb_outk` / `cb_outv` | 1/3 | `reader_create_qkv_heads_sharded.cpp` | Portable (workaround) | **undesirable but OK hack:** `reserve_back` + `get_write_ptr() + head/seq offset` scatter into reserved output region, then `push_back`; uplift: strided/multi-producer DFB on Quasar | Portable (workaround) | same |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)
