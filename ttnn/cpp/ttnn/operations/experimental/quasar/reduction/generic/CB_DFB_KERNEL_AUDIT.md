# CB→DFB Kernel Audit: `experimental/quasar/reduction/generic`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/quasar/reduction/generic/`

**Scope:** All device kernels under `device/kernels/`, legacy **and** `_metal2` variants: compute (`reduce*.cpp`, `reduce_{h,hw,w}_neg*.cpp`, `reduce_rm.cpp`, `welford_reduce_{h,hw,w}.cpp`) and dataflow (`reader_unary_reduce_rm.cpp`, `reader_unary_reduce_universal_start_id{,_metal2}.cpp`, `reader_unary_transpose_wh_*`, `reduce_rm_dataflow_common.hpp`, `writer_reduce_rm_scalar.cpp`, `writer_unary_interleaved_start_id{,_metal2}.cpp`, `writer_unary_sharded.cpp`, `writer_welford_hw.cpp`).

## Overall verdict: YELLOW

**Summary:** **Mechanical NEEDS-FIX** (three hits) away from GREEN. The RM reader (`reader_unary_reduce_rm.cpp:83`, `cb_id_rm`) and the interleaved writer — legacy (`writer_unary_interleaved_start_id.cpp:20`, `cb_id_out`) and ported (`writer_unary_interleaved_start_id_metal2.cpp:22`, `dfb::out`) — read `get_local_cb_interface(...).fifo_page_size`; getter exists (`get_entry_size()`). Notably the Welford compute kernels here use **no** `read_tile_value`/`get_tile_address` — zero QUASAR-BLOCKED hits, unlike the normalization Welford family. Zero silent-wrong / `get_pointer_to_cb_data` / `fifo_*` ptr-surgery hits. All other CBs are canonical Class 1 linear FIFOs.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_id_rm` | 1 | `reader_unary_reduce_rm.cpp:83` | Portable | **NEEDS-FIX:** `get_local_cb_interface(cb_id_rm).fifo_page_size` → `get_entry_size()` (getter exists) | Portable | same |
| `cb_id_out` / `dfb::out` | 1 | `writer_unary_interleaved_start_id.cpp:20`, `writer_unary_interleaved_start_id_metal2.cpp:22` | Portable | **NEEDS-FIX:** `get_local_cb_interface(...).fifo_page_size` → `get_entry_size()` (getter exists) | Portable | same |
| scalar / reduce-partial / welford CBs + all other in/out | 1 | `reduce*.cpp`, `welford_reduce_*.cpp`, `writer_welford_hw.cpp`, `reader_unary_transpose_wh_*`, `writer_reduce_rm_scalar.cpp`, `writer_unary_sharded.cpp`, `reduce_rm_dataflow_common.hpp` | Portable | canonical linear FIFO; scalar reduce-coeff CB is a normal FIFO; `get_read_ptr()`/`get_write_ptr()` as addr only | Portable | — |

## GATE hits (must be empty to merge)

- `device/kernels/dataflow/reader_unary_reduce_rm.cpp:83` — `get_local_cb_interface(cb_id_rm).fifo_page_size` **read** — mechanical → `get_entry_size()`.
- `device/kernels/dataflow/writer_unary_interleaved_start_id.cpp:20` — `get_local_cb_interface(cb_id_out).fifo_page_size` **read** — mechanical → `get_entry_size()`.
- `device/kernels/dataflow/writer_unary_interleaved_start_id_metal2.cpp:22` — `get_local_cb_interface(dfb::out).fifo_page_size` **read** — mechanical → `get_entry_size()`.

## Blocked on runtime (2xx rollup)

- (none) — getter exists today; no runtime API dependency. Welford compute path uses no `read_tile_value`/`get_tile_address`.
