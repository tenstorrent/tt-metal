# CB→DFB Kernel Audit: `experimental/quasar/pool_generic`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/quasar/pool_generic/`

**Scope:** All device kernels under `device/kernels/`: compute (`compute_mpwi.cpp`, `compute_pool_2d.cpp`), dataflow (`reader_mpwi.cpp`, `reader_pool_2d.cpp`), and shared header `pool_kernels_common.hpp`.

## Overall verdict: YELLOW

**Summary:** **Mechanical NEEDS-FIX** (four hits, all in `pool_kernels_common.hpp`) away from GREEN. The tile-clear/zero-fill helpers reach into `get_local_cb_interface(cb_id).fifo_num_pages` and `.fifo_page_size` to size their zero-out loops — even though the helpers already take `DataflowBuffer` arguments. Both getters exist: `fifo_num_pages` → `get_total_num_entries()` (merged, [PR #49197](https://github.com/tenstorrent/tt-metal/pull/49197)) and `fifo_page_size` → `get_entry_size()`. Zero silent-wrong / `read_tile_value` / `get_pointer_to_cb_data` / `fifo_wr_ptr`/`fifo_rd_ptr` ptr-surgery hits. The scalar-fill, MPWI, and pool-2D data paths are canonical Class 1 linear FIFOs (`reserve_back`/`push_back` + `get_write_ptr()` addr).

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb` in `clear_out_tiles`/`zero_out_tiles`/`zero_out_page` (out/intermed) | 1 | `pool_kernels_common.hpp:47,48,76,129` | Portable | **NEEDS-FIX:** `fifo_num_pages` → `get_total_num_entries()` (PR #49197); `fifo_page_size` → `get_entry_size()` (getter exists) | Portable | same |
| `cb_in` / `cb_scalar` / `cb_out` (pool data path) | 1 | `compute_mpwi.cpp`, `compute_pool_2d.cpp`, `reader_mpwi.cpp`, `reader_pool_2d.cpp`, `fill_scalar`/`load_config_tensor_if_in_dram` in `pool_kernels_common.hpp` | Portable | canonical linear FIFO; `get_read_ptr()`/`get_write_ptr()` as addr only | Portable | — |

## GATE hits (must be empty to merge)

- `device/kernels/pool_kernels_common.hpp:47` — `get_local_cb_interface(cb_id).fifo_num_pages` **read** — mechanical → `get_total_num_entries()` (PR #49197).
- `device/kernels/pool_kernels_common.hpp:48` — `get_local_cb_interface(cb_id).fifo_page_size` **read** — mechanical → `get_entry_size()`.
- `device/kernels/pool_kernels_common.hpp:76` — `get_local_cb_interface(cb_id).fifo_num_pages` **read** — mechanical → `get_total_num_entries()` (PR #49197).
- `device/kernels/pool_kernels_common.hpp:129` — `get_local_cb_interface(cb.get_id()).fifo_page_size` **read** — mechanical → `get_entry_size()`.

## Blocked on runtime (2xx rollup)

- (none) — both getters exist today (`get_total_num_entries()` merged via PR #49197); no runtime API dependency.
