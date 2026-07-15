# CB→DFB Kernel Audit: `experimental/quasar/untilize`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/quasar/untilize/`

**Scope:** All device kernels under `device/kernels/`, legacy **and** `_metal2` variants: compute (`untilize.cpp`, `untilize_metal2.cpp`, `untilize_variable_num_blocks{,_metal2}.cpp`, `untilize_w.cpp`, `untilize_wh.cpp`) and the dataflow reader/writer family (interleaved, sharded, nd-sharded, stick-layout split-rows) incl. `_metal2` variants.

## Overall verdict: YELLOW

**Summary:** One **mechanical NEEDS-FIX** away from GREEN. The interleaved reader reads `get_local_cb_interface(...).fifo_page_size` to size its NoC page reads — a GATE-shaped field read with a getter that already exists (`get_entry_size()`). Both the legacy (`reader_unary_interleaved_start_id.cpp:21`, `cb_id_in0`) and ported (`reader_unary_interleaved_start_id_metal2.cpp:22`, `dfb::in`) variants carry it. Zero silent-wrong / `read_tile_value` / `get_pointer_to_cb_data` / `fifo_*` ptr-surgery hits anywhere. All other CBs are canonical Class 1 linear FIFOs.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_id_in0` / `dfb::in` | 1 | `reader_unary_interleaved_start_id.cpp:21`, `reader_unary_interleaved_start_id_metal2.cpp:22` | Portable | **NEEDS-FIX:** `get_local_cb_interface(...).fifo_page_size` → `get_entry_size()` (getter exists) | Portable | same |
| all other CBs (`cb_in`/`cb_out`, sharded/nd/stick variants) | 1 | `untilize*.cpp` compute, `reader_unary_*`, `writer_unary_*` (+`_metal2`) | Portable | canonical linear FIFO; `get_read_ptr()`/`get_write_ptr()` as addr only | Portable | — |

## GATE hits (must be empty to merge)

- `device/kernels/dataflow/reader_unary_interleaved_start_id.cpp:21` — `get_local_cb_interface(cb_id_in0).fifo_page_size` **read** — mechanical → `get_entry_size()`.
- `device/kernels/dataflow/reader_unary_interleaved_start_id_metal2.cpp:22` — `get_local_cb_interface(dfb::in).fifo_page_size` **read** — mechanical → `get_entry_size()`.

## Blocked on runtime (2xx rollup)

- (none) — getter exists today; no runtime API dependency.
