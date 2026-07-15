# CB→DFB Kernel Audit: `experimental/quasar/sharded_to_interleaved`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/quasar/sharded_to_interleaved/`

**Scope:** All device kernels under `device/kernels/`: compute (`eltwise_copy.cpp`) and dataflow (`reader_unary_sharded.cpp`, `writer_unary_sharded_blocks_interleaved_start_id.cpp`, `writer_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp`).

## Overall verdict: GREEN

**Summary:** Clean. All Step-4 litmus scans return **zero** hits — no GATE field access, no silent-wrong pointers, no `read_tile_value`/`get_tile_address`, no `get_pointer_to_cb_data`, no `fifo_*` ptr surgery or field reads. The resident sharded input is read into a canonical FIFO, optionally copied, and drained to the interleaved output; `get_read_ptr()`/`get_write_ptr()` are used only as L1/NoC byte addresses. Note: this is the inverse of `interleaved_to_sharded`, and unlike that op it has **no** scratch-CB `fifo_page_size` read. Mechanical `CircularBuffer` → `DataflowBuffer` rename on both arches.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_in` (resident shard) | 1 | `reader_unary_sharded.cpp`, `eltwise_copy.cpp` | Portable | resident sharded input, linear FIFO → `DataflowBuffer` | Portable | — |
| `cb_out` (interleaved) | 1 | `writer_unary_sharded_blocks_interleaved_start_id.cpp`, `writer_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp` | Portable | drain → interleaved output, `get_write_ptr()` as addr only | Portable | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)
