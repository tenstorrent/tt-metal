# CB→DFB Kernel Audit: `experimental/quasar/reshard`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/quasar/reshard/`

**Scope:** All device kernels under `device/kernels/`: dataflow (`reshard_reader.cpp`, `reshard_reader_diff_width.cpp`, `reshard_same_height_reader.cpp`, `reshard_same_height_writer.cpp`, `reshard_same_width_reader.cpp`, `reshard_same_width_writer.cpp`) and nd reshard (`nd_reshard_copy_local_shards.cpp`, `nd_reshard_copy_pages_reader.cpp`, `nd_reshard_copy_pages_writer.cpp`).

## Overall verdict: GREEN

**Summary:** Clean. All Step-4 litmus scans return **zero** hits across all 9 kernels — no GATE field access, no silent-wrong pointers, no `read_tile_value`/`get_tile_address`, no `get_pointer_to_cb_data`, no `fifo_*` ptr surgery or field reads. Reshard moves shard pages between sharded CBs via NoC using canonical FIFO semantics (or direct local-shard copies); `get_read_ptr()`/`get_write_ptr()` are used only as L1/NoC byte addresses for the shard-to-shard transfers. Mechanical `CircularBuffer` → `DataflowBuffer` rename on both arches.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| sharded input/output CBs | 1 | `reshard_*_reader.cpp`, `reshard_*_writer.cpp`, `nd_reshard_copy_*` | Portable | resident shard I/O via NoC, linear FIFO; `get_read_ptr()`/`get_write_ptr()` as addr only | Portable | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)
