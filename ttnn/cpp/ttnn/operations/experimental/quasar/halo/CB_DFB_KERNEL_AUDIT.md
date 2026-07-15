# CB→DFB Kernel Audit: `experimental/quasar/halo`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/quasar/halo/`

**Scope:** All device kernels under `device/kernels/`: compute (`pack_untilize.cpp`) and dataflow (`halo_gather.cpp`).

## Overall verdict: GREEN

**Summary:** Clean. All Step-4 litmus scans return **zero** hits — no GATE field access, no silent-wrong pointers, no `read_tile_value`/`get_tile_address`, no `get_pointer_to_cb_data`, no `fifo_*` ptr surgery or field reads. The gather dataflow and pack-untilize compute stages use canonical `reserve_back`/`push_back`/`wait_front`/`pop_front`; `get_read_ptr()`/`get_write_ptr()` are used only as L1/NoC byte addresses for the halo scatter/gather NoC transfers. Mechanical `CircularBuffer` → `DataflowBuffer` rename on both arches.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_in` (sharded input / gathered halo) | 1 | `halo_gather.cpp`, `pack_untilize.cpp` | Portable | halo gather via NoC into linear FIFO; `get_write_ptr()` as addr only | Portable | — |
| `cb_out` (pack-untilize output) | 1 | `pack_untilize.cpp` | Portable | pack → output, linear FIFO | Portable | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)
