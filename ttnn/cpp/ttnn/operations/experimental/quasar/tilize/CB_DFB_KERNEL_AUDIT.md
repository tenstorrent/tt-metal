# CB→DFB Kernel Audit: `experimental/quasar/tilize`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/quasar/tilize/`

**Scope:** All device kernels under `device/kernels/`: compute (`tilize.cpp`, `tilize_wh.cpp`) and dataflow (`reader_unary_pad_multicore_both_dims_metal2.cpp`, `reader_unary_sharded.cpp`, `reader_unary_stick_layout_split_rows_{multicore,singlecore}.cpp`, `writer_unary_interleaved_start_id.cpp`, `writer_unary_interleaved_start_id_wh.cpp`, `writer_unary_sharded.cpp`).

## Overall verdict: YELLOW

**Summary:** One **mechanical NEEDS-FIX** away from GREEN. The interleaved writer reads `get_local_cb_interface(dfb::out).fifo_page_size` (`writer_unary_interleaved_start_id.cpp:16`) to size NoC page writes — getter exists (`get_entry_size()`). Zero silent-wrong / `read_tile_value` / `get_pointer_to_cb_data` / `fifo_*` ptr-surgery hits. All other CBs are canonical Class 1 linear FIFOs.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `dfb::out` (interleaved writer) | 1 | `writer_unary_interleaved_start_id.cpp:16` | Portable | **NEEDS-FIX:** `get_local_cb_interface(dfb::out).fifo_page_size` → `get_entry_size()` (getter exists) | Portable | same |
| all other CBs (`cb_in`/`cb_out`, sharded/stick/wh variants) | 1 | `tilize.cpp`, `tilize_wh.cpp`, `reader_unary_*`, `writer_unary_sharded.cpp`, `writer_unary_interleaved_start_id_wh.cpp` | Portable | canonical linear FIFO; `get_read_ptr()`/`get_write_ptr()` as addr only | Portable | — |

## GATE hits (must be empty to merge)

- `device/kernels/dataflow/writer_unary_interleaved_start_id.cpp:16` — `get_local_cb_interface(dfb::out).fifo_page_size` **read** — mechanical → `get_entry_size()`.

## Blocked on runtime (2xx rollup)

- (none) — getter exists today; no runtime API dependency.
