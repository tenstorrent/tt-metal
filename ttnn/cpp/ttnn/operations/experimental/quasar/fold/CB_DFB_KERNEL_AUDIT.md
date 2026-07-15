# CB→DFB Kernel Audit: `experimental/quasar/fold`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/quasar/fold/`

**Scope:** All device kernels under `device/kernels/`: compute (`untilize.cpp`) and dataflow (`reader_dram2cb_for_rm_input.cpp`, `reader_dram2cb_tiled.cpp`, `writer_cb2dram_for_rm_input.cpp`, `writer_cb2dram_for_tiled_input.cpp`, `writer_cb2s_row_major.cpp`).

## Overall verdict: GREEN

**Summary:** Clean. All Step-4 litmus scans return **zero** hits — no GATE field access, no silent-wrong pointers, no `read_tile_value`/`get_tile_address`, no `get_pointer_to_cb_data`, no `fifo_*` ptr surgery or field reads. DRAM→CB readers, the untilize compute stage, and CB→DRAM/CB→shard writers are all canonical linear FIFOs; `get_read_ptr()`/`get_write_ptr()` serve only as L1/NoC byte addresses. Mechanical `CircularBuffer` → `DataflowBuffer` rename on both arches.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_in` (DRAM→CB) | 1 | `reader_dram2cb_for_rm_input.cpp`, `reader_dram2cb_tiled.cpp` | Portable | staged input, linear FIFO → `DataflowBuffer` | Portable | — |
| `cb_out` (CB→DRAM/shard) | 1 | `untilize.cpp`, `writer_cb2dram_*.cpp`, `writer_cb2s_row_major.cpp` | Portable | pack/drain, `get_write_ptr()` as L1/NoC addr only | Portable | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)
