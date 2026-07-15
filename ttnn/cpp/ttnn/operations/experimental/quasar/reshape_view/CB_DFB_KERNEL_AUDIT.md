# CB→DFB Kernel Audit: `experimental/quasar/reshape_view`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/quasar/reshape_view/`

**Scope:** Device kernels for the reshape op. Note: kernels live under the spurious nested path `reshape_view/device/device/` — treated as ONE op with a single doc here. Kernels: `device/device/dataflow/reader_reshape_tiled.cpp`, `device/device/dataflow/writer_reshape_tiled.cpp`, `device/device/rm_reshape_interleaved.cpp`.

## Overall verdict: GREEN

**Summary:** Clean. All Step-4 litmus scans return **zero** hits across the tiled reader/writer and the row-major interleaved reshape kernel — no GATE field access, no silent-wrong pointers, no `read_tile_value`/`get_tile_address`, no `get_pointer_to_cb_data`, no `fifo_*` ptr surgery or field reads. Reshape streams pages/tiles through canonical linear FIFOs; `get_read_ptr()`/`get_write_ptr()` are used only as L1/NoC byte addresses. Mechanical `CircularBuffer` → `DataflowBuffer` rename on both arches.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_in` / `cb_out` (reshape stream) | 1 | `reader_reshape_tiled.cpp`, `writer_reshape_tiled.cpp`, `rm_reshape_interleaved.cpp` | Portable | tiled + row-major reshape, linear FIFO; `get_read_ptr()`/`get_write_ptr()` as addr only | Portable | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)
