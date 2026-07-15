# CB→DFB Kernel Audit: `kv_cache`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/kv_cache/`

**Scope:**

- Factory `UpdateCache` (`update_cache_multi_core_program_factory.cpp`) → `reader_update_cache_interleaved_start_id.cpp`, `writer_update_cache_interleaved_start_id.cpp`, `compute/update_cache.cpp`
- Factory `FillCache` (`fill_cache_multi_core_program_factory.cpp`) → `reader_fill_cache_interleaved_start_id.cpp`, **donor** `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`

## Overall verdict: RED

**Summary:** kv_cache's **own** kernels are clean (Class 1 + WEIRD-OK in-place update; only `get_read_ptr`/`get_write_ptr`/`get_tile_size`). The only GATE is inherited from the cross-op **donor** writer used by the FillCache factory — a mechanical `fifo_page_size` → `get_entry_size()` swap owned by `eltwise/unary`. The UpdateCache factory is GREEN on its own.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cache_cb` | 1 | `reader_update_cache.cpp`, `writer_update_cache.cpp`, `compute/update_cache.cpp` | Portable | reader fills → compute untilizes / writer NOC-writes tilized result; `get_tile_size()`, ptr cursors | Portable | — |
| `input_cb` | 1 | `reader_update_cache.cpp`, `compute/update_cache.cpp` | Portable | reader → compute untilize, linear FIFO | Portable | — |
| `untilized_in_cb` | 1 | `compute/update_cache.cpp`, `writer_update_cache.cpp` | Portable | compute produce → writer `get_read_ptr()` NOC source | Portable | — |
| `untilized_cache_cb` | 4/5 | `compute/update_cache.cpp`, `writer_update_cache.cpp` | Portable (workaround) | **undesirable but OK hack:** in-place RMW — writer NOC-reads new K/V into `get_read_ptr()+offset` of compute's untilized block | Portable (workaround) | uplift: scratchpad+sems or LTA (borrowed) |
| `untilized_cache2_cb` | 4 | `writer_update_cache.cpp`, `compute/update_cache.cpp` | Portable (workaround) | **undesirable but OK hack:** aliases `untilized_cache_cb` backing; writer `push_back` → compute retilizes (host aliasing/`borrowed_from` concern, not kernel field access) | Portable (workaround) | same |
| `out_cb` | 1 | `compute/update_cache.cpp` | Portable | compute tilize pack → drained via `cache_cb` write path | Portable | — |
| `src0_cb` (c_0, FillCache) | 1 | `reader_fill_cache.cpp`, **donor** `writer_unary_interleaved_start_id.cpp` | Blocked | linear FIFO (`get_write_ptr()` cursor); GATE in **donor**: `writer_unary_interleaved_start_id.cpp:19` `.fifo_page_size` → `get_entry_size()` | Blocked | same |

## GATE hits (must be empty to merge)

- `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp:19` (donor, FillCache) — `get_local_cb_interface(cb_id_out).fifo_page_size` — → `dfb.get_entry_size()`

## Blocked on runtime (2xx rollup)

- (none)
