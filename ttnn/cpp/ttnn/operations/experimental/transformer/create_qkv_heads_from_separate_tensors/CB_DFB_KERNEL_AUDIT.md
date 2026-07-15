# CB→DFB Kernel Audit: `create_qkv_heads_from_separate_tensors`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/transformer/create_qkv_heads_from_separate_tensors/`

**Scope:** `device/kernels/reader_create_qkv_heads_sharded_separate.cpp`. Single sharded reader; no shared donor kernels beyond `api/` headers.

## Overall verdict: GREEN

**Summary:** Same head-split reshuffle as `create_qkv_heads`, but Q comes from `cb_inq` and fused KV from `cb_inkv`. Output shards `cb_outq`/`cb_outk`/`cb_outv` are filled via `reserve_back` → `get_write_ptr() + byte_offset` scatter → `push_back`; inputs read via `get_read_ptr()` NoC source. Step-4 litmus scans return **zero** hits (no field access, no ptr surgery, no `get_pointer_to_cb_data`).

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_inq` | 1 | `reader_create_qkv_heads_sharded_separate.cpp` | Portable | Q input shard; `get_read_ptr()` as NoC source | Portable | — |
| `cb_inkv` | 1 | `reader_create_qkv_heads_sharded_separate.cpp` | Portable | fused KV input shard; `get_read_ptr()` as NoC source | Portable | — |
| `cb_outq` / `cb_outk` / `cb_outv` | 1/3 | `reader_create_qkv_heads_sharded_separate.cpp` | Portable (workaround) | **undesirable but OK hack:** `reserve_back` + `get_write_ptr() + head/seq offset` scatter into reserved output region, then `push_back`; uplift: strided/multi-producer DFB on Quasar | Portable (workaround) | same |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)
