# CB→DFB Kernel Audit: `nlp_concat_heads_decode`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode/`

**Scope:** `device/kernels/dataflow/reader_tm_tile_layout_nlp_concat_heads_decode.cpp`, `device/kernels/dataflow/reader_tm_tile_layout_nlp_concat_heads_decode_subcoregrid.cpp`. No shared donor kernels beyond `api/` headers.

## Overall verdict: GREEN

**Summary:** Decode-mode head concat. The reader gathers per-batch head slices over the NoC and lands them into `cb_q_out` at computed per-batch byte offsets (`get_write_ptr() + wptr_offset`) inside the reserved output region. This is a standard sharded-output scatter via bare `get_write_ptr()` — no `fifo_*` field access, no `get_local_cb_interface`, no `get_pointer_to_cb_data`. Step-4 litmus scans return **zero** hits.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_id_q_out` | 1/3 | `reader_*_decode.cpp`, `reader_*_decode_subcoregrid.cpp` | Portable (workaround) | **undesirable but OK hack:** `get_write_ptr() + per-batch offset` scatter into reserved output shard; uplift: strided/multi-producer DFB on Quasar | Portable (workaround) | same |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)
