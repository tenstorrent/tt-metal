# CB→DFB Kernel Audit: `all_gather_minimal_matmul_async`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async/`

**Scope:** All in-scope device kernels under `device/kernels/`: `compute.cpp`, `dm_in0_sender.cpp`, `dm_in1_sender_out.cpp`, and shared header `matmul_dataflow_common.hpp`.

## Overall verdict: GREEN

**Summary:** Fused all-gather + minimal matmul with its own self-contained kernels (not the legacy `bmm_*_gathered` compute). All Step-4 litmus scans return **zero** hits — no GATE, no silent-wrong, no `read_tile_value`/`get_tile_address`, no `get_pointer_to_cb_data`, no ptr surgery, no `fifo_*` field reads. The compute kernel is a canonical matmul tile pipeline (41 FIFO sync ops), and the DM senders use canonical FIFO + bare `get_read_ptr()`/`get_write_ptr()` for NoC. Mechanical rename.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_in0` / `cb_id_in0` | 1 | `compute.cpp`, `dm_in0_sender.cpp`, `matmul_dataflow_common.hpp` | Portable | activation input, canonical FIFO | Portable | — |
| `cb_in1` / `cb_id_in1`, `cb_in2` / `cb_id_in2` | 1 | `compute.cpp`, `dm_in1_sender_out.cpp`, `matmul_dataflow_common.hpp` | Portable | weights / gathered input, linear FIFO | Portable | — |
| `cb_out` / `cb_id_out` (`out_cb_id`, `intermediate_cb_id`) | 1 | `compute.cpp`, `dm_in1_sender_out.cpp` | Portable | pack → output, `get_write_ptr()` as L1/NoC addr | Portable | — |
| `cb_ternary_a` / `cb_ternary_b` (`ternary_*_cb_id`) | 1 | `matmul_dataflow_common.hpp` | Portable | optional fused ternary inputs, linear FIFO | Portable | — |
| `reserved_packet_header_cb_id` / `cb_packet_header` | 1/6 | `dm_in0_sender.cpp`, `dm_in1_sender_out.cpp` | Portable | fabric packet-header scratch; `reserve_back` + `get_write_ptr()` | Portable | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)

## Recommended path

Port freely — mechanical `CircularBuffer` → `DataflowBuffer` rename. This op's fused compute is a clean canonical matmul (contrast `llama_all_gather_matmul_async`, which reuses the RED `bmm_large_block_zm_fused_bias_activation_gathered.cpp`).
