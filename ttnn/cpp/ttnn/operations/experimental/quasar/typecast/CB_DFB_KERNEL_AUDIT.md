# CB→DFB Kernel Audit: `experimental/quasar/typecast`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/quasar/typecast/`

**Scope:** All device kernels under `device/kernels/`: compute (`eltwise_typecast.cpp`) and dataflow (`reader_typecast_rm_chunked.cpp`, `reader_unary_interleaved_start_id.cpp`, `reader_unary_sharded.cpp`, `writer_typecast_rm_chunked.cpp`, `writer_unary_interleaved_start_id.cpp`).

## Overall verdict: YELLOW

**Summary:** One **mechanical NEEDS-FIX** (two hits) away from GREEN. Both the interleaved reader (`reader_unary_interleaved_start_id.cpp:21`, `cb_id_in0`) and writer (`writer_unary_interleaved_start_id.cpp:20`, `cb_id_out`) read `get_local_cb_interface(...).fifo_page_size` to size NoC page transfers — getter exists (`get_entry_size()`). Zero silent-wrong / `read_tile_value` / `get_pointer_to_cb_data` / `fifo_*` ptr-surgery hits. All other CBs are canonical Class 1 linear FIFOs.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_id_in0` | 1 | `reader_unary_interleaved_start_id.cpp:21` | Portable | **NEEDS-FIX:** `get_local_cb_interface(cb_id_in0).fifo_page_size` → `get_entry_size()` (getter exists) | Portable | same |
| `cb_id_out` | 1 | `writer_unary_interleaved_start_id.cpp:20` | Portable | **NEEDS-FIX:** `get_local_cb_interface(cb_id_out).fifo_page_size` → `get_entry_size()` (getter exists) | Portable | same |
| all other CBs (`cb_in`/`cb_out`, rm-chunked, sharded variants) | 1 | `eltwise_typecast.cpp`, `reader_typecast_rm_chunked.cpp`, `reader_unary_sharded.cpp`, `writer_typecast_rm_chunked.cpp` | Portable | canonical linear FIFO; `get_read_ptr()`/`get_write_ptr()` as addr only | Portable | — |

## GATE hits (must be empty to merge)

- `device/kernels/dataflow/reader_unary_interleaved_start_id.cpp:21` — `get_local_cb_interface(cb_id_in0).fifo_page_size` **read** — mechanical → `get_entry_size()`.
- `device/kernels/dataflow/writer_unary_interleaved_start_id.cpp:20` — `get_local_cb_interface(cb_id_out).fifo_page_size` **read** — mechanical → `get_entry_size()`.

## Blocked on runtime (2xx rollup)

- (none) — getter exists today; no runtime API dependency.
