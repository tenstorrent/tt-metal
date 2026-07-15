# CB→DFB Kernel Audit: `zero_padded_kv_cache`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/zero_padded_kv_cache/`

**Scope:** In-scope device kernels under `device/kernels/` → `compute/zero_padded_kv_cache.cpp`, `dataflow/reader_zero_padded_kv_cache.cpp`, `dataflow/writer_zero_padded_kv_cache.cpp`, `zero_padded_kv_cache_common.hpp` (shared per-chip pad-window helper, no CB access).

## Overall verdict: GREEN

**Summary:** No GATE, no silent-wrong, no ptr surgery, no field reads, no runtime blockers. The `src`, `mask`, and `out` CBs are canonical Class 1 linear FIFOs (`reserve_back`/`push_back` producer → `wait_front`/`pop_front` consumer); `get_write_ptr()` is used only as an L1 byte address when building the row-mask tile. The `zero` CB is a Class 6 private zeroing scratch — `reserve_back(1)` with **no** `push_back` (read ptr == just-zeroed write ptr for the DRAM `async_write_zeros` overload); its clean end-state is `ScratchpadSpec` (autoportable), but it involves no `LocalCBInterface` access so it ports today.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `src_cb` | 1 | `reader_zero_padded_kv_cache.cpp`, `compute/zero_padded_kv_cache.cpp` | Portable | boundary partial tile read; canonical `reserve/push`·`wait/pop` linear FIFO | Portable | — |
| `mask_cb` | 1 | `reader_zero_padded_kv_cache.cpp`, `compute/zero_padded_kv_cache.cpp` | Portable | bf16 row-mask built in L1 via `get_write_ptr()` (L1 addr only), then `push_back` → compute | Portable | — |
| `out_cb` | 1 | `compute/zero_padded_kv_cache.cpp`, `writer_zero_padded_kv_cache.cpp` | Portable | masked partial tile pack → writer, canonical `wait_front`/`pop_front` | Portable | — |
| `zero_cb` | 6 | `writer_zero_padded_kv_cache.cpp` | Portable | private zeroing scratch: `reserve_back(1)` never `push_back`'d; `async_write_zeros(zero, ...)`. Clean end-state **ScratchpadSpec** (autoportable); no `LocalCBInterface` access → ports today | Portable | same |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)
