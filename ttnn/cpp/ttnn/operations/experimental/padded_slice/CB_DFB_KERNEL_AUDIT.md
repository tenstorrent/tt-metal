# CB→DFB Kernel Audit: `padded_slice`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/padded_slice/`

**Scope:** `padded_slice_rm_program_factory.cpp` + `padded_slice_tile_program_factory.cpp` → kernels: `dataflow/padded_slice_reader_rm_interleaved_start_id.cpp`, `dataflow/padded_slice_reader_tiled_interleaved_start_id.cpp`, `dataflow/writer_unary_sharded_padded_rm.cpp`, `dataflow/writer_unary_sharded_padded_tiled.cpp`; **donor** kernels `sliding_window/halo/device/kernels/compute/pack_untilize.cpp`, `data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp`; donor header `data_movement/common/kernels/common.hpp`.

## Overall verdict: YELLOW

**Summary:** All CBs are canonical Class 1 FIFOs / scratch-staging buffers. The only findings are two `fifo_page_size` reads in `padded_slice_reader_rm_interleaved_start_id.cpp` (both **mechanical NEEDS-FIX** → `get_entry_size()`, getter exists today). Line 85 is inside a `#ifdef DEBUG` `DPRINT`; line 97 is a live read used to compute per-slot offsets in the TRID scratch region. Both clear the GATE with a trivial getter swap; nothing is runtime-blocked. Donor kernels (`pack_untilize.cpp`, `writer_unary_sharded.cpp`) and `common.hpp` are clean.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_id_in0` (input) | 1 | `padded_slice_reader_rm_interleaved_start_id.cpp`, `padded_slice_reader_tiled_interleaved_start_id.cpp` | Portable | **NEEDS-FIX:** debug-only `get_local_cb_interface(cb_id_in0).fifo_page_size` (`reader_rm:85`, inside `#ifdef DEBUG` `DPRINT`) → `get_entry_size()`. Canonical `reserve_back` + `get_write_ptr()` reader otherwise. | Portable | same |
| `cb_id_non_aligned` (TRID scratch) | 2/6 | `padded_slice_reader_rm_interleaved_start_id.cpp` | Portable | **NEEDS-FIX:** `get_local_cb_interface(cb_id_non_aligned).fifo_page_size` (`reader_rm:97`) → `get_entry_size()`. Private src→scratch→dst TRID staging (`reserve_back(num_trids)` + per-slot `get_write_ptr()` offsets); getter swap clears GATE, safe to port. Optional uplift: `ScratchpadSpec`. | Portable | same |
| output / halo compute CBs | 1 | `writer_unary_sharded_padded_rm.cpp`, `writer_unary_sharded_padded_tiled.cpp`, `pack_untilize.cpp` (donor), `writer_unary_sharded.cpp` (donor) | Portable | pack/untilize + sharded output, canonical FIFO + bare `get_write_ptr()` L1 addr | Portable | — |

## GATE hits (must be empty to merge)

- `padded_slice/device/kernels/dataflow/padded_slice_reader_rm_interleaved_start_id.cpp:85` — `get_local_cb_interface(cb_id_in0).fifo_page_size` **read** (inside `#ifdef DEBUG` `DPRINT`) — **mechanical**: → `get_entry_size()`.
- `padded_slice/device/kernels/dataflow/padded_slice_reader_rm_interleaved_start_id.cpp:97` — `get_local_cb_interface(cb_id_non_aligned).fifo_page_size` **read** — **mechanical**: → `get_entry_size()`.

## Blocked on runtime (2xx rollup)

- (none)
