# CB→DFB Kernel Audit: `all_to_all_dispatch_metadata`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/ccl/all_to_all_dispatch_metadata/`

**Scope:** All in-scope device kernels under `device/kernels/`: `dataflow/reader_all_to_all_dispatch_metadata.cpp`, `dataflow/writer_all_to_all_dispatch_metadata.cpp`.

## Overall verdict: GREEN

**Summary:** MoE dispatch-metadata all-to-all = NoC/semaphore movement of expert indices/mapping/scores (no compute). All Step-4 litmus scans return **zero** hits — no GATE, no silent-wrong, no `read_tile_value`/`get_tile_address`, no `get_pointer_to_cb_data`, no ptr surgery, no `fifo_*` field reads. All CBs are canonical linear-FIFO metadata buffers plus a reserved fabric packet-header scratch CB, addressed via bare `get_read_ptr()`/`get_write_ptr()`. Mechanical rename. (Note: this is `all_to_all_dispatch_metadata` — not the out-of-scope `deepseek_moe_gate` / `generalized_moe_gate` firmware-reconfig path.)

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_input` (`input_tensor_cb_id`), `cb_indices` (`indices_tensor_cb_id`), `cb_mapping` (`mapping_tensor_cb_id`), `cb_scores` (`scores_tensor_cb_id`), `cb_metadata` | 1 | `dataflow/reader_all_to_all_dispatch_metadata.cpp`, `dataflow/writer_all_to_all_dispatch_metadata.cpp` | Portable | canonical FIFO metadata streams; `get_read_ptr()`/`get_write_ptr()` as L1/NoC addr | Portable | — |
| `cb_send_preparation_buffer` (`send_preparation_buffer_cb_id`) | 1/6 | `dataflow/reader_all_to_all_dispatch_metadata.cpp`, `dataflow/writer_all_to_all_dispatch_metadata.cpp` | Portable | send-staging scratch; canonical `reserve_back` + `get_write_ptr()`, no field access | Portable | — |
| `reserved_packet_header_cb_id` / `cb_packet_header` (`packet_header_cb_id`) | 1/6 | both kernels | Portable | fabric packet-header scratch; `reserve_back` + `get_write_ptr()` | Portable | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)

## Recommended path

Port freely — mechanical `CircularBuffer` → `DataflowBuffer` rename. No field surgery, no runtime API dependency, no LTA prerequisite.
