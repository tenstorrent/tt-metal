# CB→DFB Kernel Audit: `experimental/quasar/tilize_with_val_padding`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/quasar/tilize_with_val_padding/`

**Scope:** All device kernels under `device/kernels/`, legacy **and** `_metal2` variants: compute (`tilize.cpp`, `tilize_metal2.cpp`, `tilize_wh.cpp`) and dataflow (`reader_unary_pad_dims_split_rows{,_multicore}.cpp`, `reader_unary_pad_height_width_sharded.cpp`, `reader_unary_pad_multicore_both_dims.cpp`, `writer_unary_interleaved_start_id{,_metal2,_wh}.cpp`, `writer_unary_sharded.cpp`).

## Overall verdict: YELLOW

**Summary:** One **mechanical NEEDS-FIX** (two hits) away from GREEN. The interleaved writer reads `get_local_cb_interface(...).fifo_page_size` to size NoC page writes in both the legacy (`writer_unary_interleaved_start_id.cpp:20`, `cb_id_out`) and ported (`writer_unary_interleaved_start_id_metal2.cpp:22`, `dfb::out`) variants — getter exists (`get_entry_size()`). Zero silent-wrong / `read_tile_value` / `get_pointer_to_cb_data` / `fifo_*` ptr-surgery hits. All other CBs are canonical Class 1 linear FIFOs.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_id_out` / `dfb::out` | 1 | `writer_unary_interleaved_start_id.cpp:20`, `writer_unary_interleaved_start_id_metal2.cpp:22` | Portable | **NEEDS-FIX:** `get_local_cb_interface(...).fifo_page_size` → `get_entry_size()` (getter exists) | Portable | same |
| all other CBs (`cb_in`/`cb_out`, pad-value, sharded/wh variants) | 1 | `tilize{,_metal2,_wh}.cpp`, `reader_unary_pad_*`, `writer_unary_sharded.cpp`, `writer_unary_interleaved_start_id_wh.cpp` | Portable | canonical linear FIFO; `get_read_ptr()`/`get_write_ptr()` as addr only | Portable | — |

## GATE hits (must be empty to merge)

- `device/kernels/dataflow/writer_unary_interleaved_start_id.cpp:20` — `get_local_cb_interface(cb_id_out).fifo_page_size` **read** — mechanical → `get_entry_size()`.
- `device/kernels/dataflow/writer_unary_interleaved_start_id_metal2.cpp:22` — `get_local_cb_interface(dfb::out).fifo_page_size` **read** — mechanical → `get_entry_size()`.

## Blocked on runtime (2xx rollup)

- (none) — getter exists today; no runtime API dependency.
