# CB→DFB Kernel Audit: `update_padded_kv_cache`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/`

**Scope:** In-scope device kernels under `device/kernels/` → `dataflow/reader_update_padded_kv_cache.cpp`, `dataflow/writer_update_padded_kv_cache.cpp`.

## Overall verdict: YELLOW

**Summary:** Both CBs are canonical Class 1 linear FIFOs. The only issue is a single **mechanical** GATE field read: `writer_update_padded_kv_cache.cpp:66` reads `get_local_cb_interface(cb_id_out).fifo_page_size` to get the per-page byte size for the NoC write. The getter exists today (`get_entry_size()`), so this is a one-line NEEDS-FIX that clears the GATE with no runtime dependency. No silent-wrong, no ptr surgery, no LTA prereq, no runtime blocker.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_id_in0` | 1 | `reader_update_padded_kv_cache.cpp` | Portable | input read into cache-update CB, canonical linear FIFO | Portable | — |
| `cb_id_out` | 1 | `writer_update_padded_kv_cache.cpp` | Portable | **NEEDS-FIX:** `writer_update_padded_kv_cache.cpp:66` `get_local_cb_interface(cb_id_out).fifo_page_size` → `cb.get_entry_size()` (getter exists). Otherwise canonical `wait_front`/`pop_front`, `noc.async_write(cb, ...)`. | Portable | same |

## GATE hits (must be empty to merge)

- `dataflow/writer_update_padded_kv_cache.cpp:66` — `get_local_cb_interface(cb_id_out).fifo_page_size` **read** — **mechanical**: → `cb.get_entry_size()` (getter exists today, no runtime dependency). Clears the GATE with a one-line swap.

## Blocked on runtime (2xx rollup)

- (none)
