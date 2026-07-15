# CB→DFB Kernel Audit: `prefix_scan`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/ssm/prefix_scan/`

**Scope:** `prefix_scan_program_factory` → kernels: `device/kernels/reader_ssm_prefix_scan.cpp`, `device/kernels/ssm_prefix_scan.cpp`, `device/kernels/writer_ssm_prefix_scan.cpp`.

## Overall verdict: GREEN

**Summary:** SSM prefix-scan pipeline (reader → tilize/scan compute → writer) with a recurrent hidden-state accumulator. Litmus scans find **zero** `get_local_cb_interface`/`cb_interface.` access, **zero** `get_cb_tiles_*_ptr`, **zero** `read_tile_value`/`get_tile_address`, **zero** `get_pointer_to_cb_data`, and **zero** `fifo_*` surgery or field reads. All CBs are canonical Class 1 linear FIFO. The reader's `CoreLocalMem<...>(cb_in.get_write_ptr() + bytes_to_copy)` pad writes are bare L1 addressing into the CB's reserved region (WEIRD-OK / already portable). Mechanical `CircularBuffer` → `DataflowBuffer` rename only.

## CB portability

CBs collapsed by role (`_obj`/`_id` aliases are the same buffer's handle / CT-arg id).

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_a`, `cb_bx`, `cb_h_in` (+ `*_tilize_in`) | 1 | `reader_ssm_prefix_scan.cpp`, `ssm_prefix_scan.cpp` | Portable (workaround) | linear FIFO inputs; **undesirable but OK hack:** reader pads via `get_write_ptr() + bytes_to_copy` (bare L1 addressing into reserved region, not `fifo_*` surgery) | Portable (workaround) | same |
| `cb_h_prev`, `cb_h`, `cb_h_acc`, `cb_ah` | 1 | `ssm_prefix_scan.cpp` | Portable | recurrent hidden-state / scan intermediates, canonical `reserve/push` ↔ `wait/pop` | Portable | — |
| `cb_tilize_out` | 1 | `ssm_prefix_scan.cpp` | Portable | tilize intermediate, linear FIFO | Portable | — |
| `cb_out` | 1 | `ssm_prefix_scan.cpp`, `writer_ssm_prefix_scan.cpp` | Portable | pack → output, `get_read_ptr()`/`get_write_ptr()` as L1/NoC addr only | Portable | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)
