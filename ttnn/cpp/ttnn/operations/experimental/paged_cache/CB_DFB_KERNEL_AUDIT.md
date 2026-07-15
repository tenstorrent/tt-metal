# CB→DFB Kernel Audit: `paged_cache`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/paged_cache/`

**Scope:** All three device sub-factories, combined:
- **update_cache** (`update_cache/paged_update_cache_program_factory.cpp`) → `dataflow/reader_update_cache_interleaved_start_id.cpp`, `dataflow/writer_update_cache_interleaved_start_id.cpp`, `compute/update_cache.cpp`
- **fused_update_cache** (tiled + row-major) → `dataflow/reader_paged_fused_update_cache_interleaved_start_id.cpp`, `dataflow/writer_paged_fused_update_cache_interleaved_start_id.cpp`, `compute/paged_fused_update_cache.cpp`, `dataflow/reader_paged_row_major_fused_update_cache_interleaved_start_id.cpp`, `dataflow/writer_paged_row_major_fused_update_cache_interleaved_start_id.cpp`, `compute/paged_row_major_fused_update_cache.cpp`
- **fill_cache** (`fill_cache/paged_fill_cache_program_factory.cpp`) → `dataflow/reader_fill_cache_interleaved.cpp`, `dataflow/writer_fill_cache_interleaved.cpp`

## Overall verdict: GREEN

**Summary:** Across all three sub-factories, every CB is a canonical Class 1 linear FIFO / staging buffer via the modern `CircularBuffer` object API. Step-4 litmus scans return **zero** hits over the full kernel set — no GATE, no silent-wrong, no ptr surgery, no field reads, no runtime-blocked APIs. The `untilized_*` intermediates are canonical tilize/untilize staging FIFOs (no ptr surgery). Mechanical `CircularBuffer` → `DataflowBuffer` rename only.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cache_cb` | 1 | all readers/writers/compute | Portable | cache tiles in/out, canonical FIFO | Portable | — |
| `in_cb` / `in1_cb` / `in2_cb` | 1 | update/fused readers + compute | Portable | new K/V input tiles, linear FIFO | Portable | — |
| `out_cb` | 1 | compute + writers | Portable | pack → cache output, `get_write_ptr()` L1/NoC addr | Portable | — |
| `untilized_cache_cb`, `untilized_cache2_cb`, `untilized_in_cb` | 1 | fused/update compute + writers | Portable | tilize/untilize staging FIFO (no ptr/field surgery) | Portable | — |
| `cb_index`, `cb_page_table` | 1 | readers/writers | Portable | update index / page-table lookup tiles, canonical FIFO | Portable | — |
| `cb_batch_idx` | 1 | `writer_fill_cache_interleaved.cpp` | Portable | fill-cache batch index tile, linear FIFO | Portable | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)
