# CB→DFB Kernel Audit: `experimental/quasar/interleaved_to_sharded`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/quasar/interleaved_to_sharded/`

**Scope:** All device kernels under `device/kernels/`: compute (`eltwise_copy.cpp`) and dataflow (`reader_unary_sharded_blocks_interleaved_start_id.cpp`, `reader_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp`, `writer_unary_sharded.cpp`, `writer_unary_sharded_blocks_start_id.cpp`, `writer_unary_sharded_stick_layout_start_id.cpp`).

## Overall verdict: YELLOW

**Summary:** One **mechanical NEEDS-FIX** away from GREEN. The stick-layout reader uses a **scratch CB** (`dfb::in1`) as a private staging region for the src→scratch→dest trid pipeline: it `reserve_back(num_trids)` then reads `get_local_cb_interface(dfb::in1).fifo_page_size` (`reader_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp:54`) to compute per-slot scratch offsets. The field read is mechanical (getter exists: `get_entry_size()`). The ideal end-state for this scratch region is `ScratchpadSpec`, but the getter swap alone clears the GATE and is safe to port. Zero silent-wrong / `read_tile_value` / `get_pointer_to_cb_data` / `fifo_wr_ptr`/`fifo_rd_ptr` ptr-surgery hits. All other CBs are canonical Class 1 linear FIFOs.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `dfb::in1` (scratch) | 2/6 | `reader_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp:54` | Portable | **NEEDS-FIX:** `get_local_cb_interface(dfb::in1).fifo_page_size` → `get_entry_size()` (getter exists). Private scratch region (`reserve_back(num_trids)` + per-slot `get_write_ptr()` offsets); ideal end-state `ScratchpadSpec`, but getter swap alone clears the GATE. | Portable | same |
| `cb_in` (resident interleaved input) | 1 | `reader_unary_sharded_blocks_interleaved_start_id.cpp`, `eltwise_copy.cpp` | Portable | linear FIFO → `DataflowBuffer`; `get_read_ptr()` as addr only | Portable | — |
| `cb_out` (sharded output) | 1 | `writer_unary_sharded.cpp`, `writer_unary_sharded_blocks_start_id.cpp`, `writer_unary_sharded_stick_layout_start_id.cpp` | Portable | drain → resident shard, `get_write_ptr()` as addr only | Portable | — |

## GATE hits (must be empty to merge)

- `device/kernels/dataflow/reader_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp:54` — `get_local_cb_interface(dfb::in1).fifo_page_size` **read** — mechanical → `get_entry_size()`.

## Blocked on runtime (2xx rollup)

- (none) — getter exists today; no runtime API dependency.
