# CB→DFB Kernel Audit: `bcast_to`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/bcast_to/`

**Scope:** program factory → kernels (4 broadcast variants): `device/kernels/dataflow/reader_interleaved_{col,row,scalar,no}_bcast_to.cpp`, `device/kernels/compute/compute_interleaved_{col,row,scalar,no}_bcast_to.cpp`, `device/kernels/dataflow/writer_interleaved_{col,row,scalar,no}_bcast_to.cpp`.

## Overall verdict: GREEN

**Summary:** Broadcast-to op with four reader/compute/writer variants (col, row, scalar, none). Litmus scans across all 12 kernels find **zero** `get_local_cb_interface`/`cb_interface.` access, **zero** `get_cb_tiles_*_ptr`, **zero** `read_tile_value`/`get_tile_address`, **zero** `get_pointer_to_cb_data`, and **zero** `fifo_*` surgery or field reads. All CBs are canonical Class 1 linear FIFO. Mechanical `CircularBuffer` → `DataflowBuffer` rename only.

## CB portability

CBs collapsed by role (`cb_id_*` aliases are the CT-arg id of the same buffer); identical across all four variants.

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_src` | 1 | `reader_interleaved_*_bcast_to.cpp`, `compute_interleaved_*_bcast_to.cpp` | Portable | input tiles, linear FIFO → `DataflowBuffer` | Portable | — |
| `cb_dst` | 1 | `compute_interleaved_*_bcast_to.cpp`, `writer_interleaved_*_bcast_to.cpp` | Portable | pack → output, `get_read_ptr()`/`get_write_ptr()` as L1/NoC addr only | Portable | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)
