# CB→DFB Kernel Audit: `experimental/quasar/untilize_with_unpadding`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/quasar/untilize_with_unpadding/`

**Scope:** All device kernels under `device/kernels/`: compute (`eltwise_copy.cpp`, `untilize.cpp`, `untilize_metal2.cpp`, `untilize_variable_num_blocks.cpp`, `untilize_w.cpp`, `untilize_wh.cpp`) and the dataflow reader/writer family (interleaved col/wh multicore, nd-sharded, sharded, stick-layout unpad variants).

## Overall verdict: YELLOW

**Summary:** One **mechanical NEEDS-FIX** away from GREEN. The interleaved reader reads `get_local_cb_interface(dfb::in).fifo_page_size` (`reader_unary_interleaved_start_id.cpp:16`) to size NoC page reads — getter exists (`get_entry_size()`). Zero silent-wrong / `read_tile_value` / `get_pointer_to_cb_data` / `fifo_*` ptr-surgery hits. (A `fifo_limit`/`fifo_rd_ptr` mention in `factories/..._program_factory.cpp` is a host-side **comment**, not a kernel field access.) All other CBs are canonical Class 1 linear FIFOs.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `dfb::in` (interleaved reader) | 1 | `reader_unary_interleaved_start_id.cpp:16` | Portable | **NEEDS-FIX:** `get_local_cb_interface(dfb::in).fifo_page_size` → `get_entry_size()` (getter exists) | Portable | same |
| all other CBs (`cb_in`/`cb_out`, sharded/nd/stick/col-multicore variants) | 1 | `untilize*.cpp`, `eltwise_copy.cpp`, `reader_unary_*`, `writer_unary_*` | Portable | canonical linear FIFO; `get_read_ptr()`/`get_write_ptr()` as addr only | Portable | — |

## GATE hits (must be empty to merge)

- `device/kernels/dataflow/reader_unary_interleaved_start_id.cpp:16` — `get_local_cb_interface(dfb::in).fifo_page_size` **read** — mechanical → `get_entry_size()`.

## Blocked on runtime (2xx rollup)

- (none) — getter exists today; no runtime API dependency.
