# CB→DFB Kernel Audit: `llama_reduce_scatter_create_heads`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter_create_heads/`

**Scope:** All in-scope device kernels under `device/kernels/`: `compute/reduction.cpp`, `dataflow/reader_llama_reduce_scatter.cpp`, `dataflow/writer_llama_reduce_scatter.cpp`.

## Overall verdict: GREEN

**Summary:** Llama reduce-scatter fused with create-heads = fabric NoC/semaphore movement + a canonical reduction compute kernel, with head-split placement in the writer. All Step-4 litmus scans return **zero** hits — no GATE, no silent-wrong, no `read_tile_value`/`get_tile_address`, no `get_pointer_to_cb_data`, no ptr surgery, no `fifo_*` field reads. CBs are canonical linear-FIFO input/output/accumulator, fabric sender/receiver staging, and a packet-header scratch CB; head placement uses bare `get_write_ptr()` + byte offsets. Mechanical rename.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_input_tensor` (`input_tensor_cb_id`) | 1 | `dataflow/reader_llama_reduce_scatter.cpp`, `compute/reduction.cpp` | Portable | input tiles, canonical FIFO | Portable | — |
| `cb_accumulator` (`accumulator_cb_id`) | 1 | `compute/reduction.cpp` | Portable | reduction accumulator via canonical FIFO — **no** rd/wr ptr surgery | Portable | — |
| output CB (`output_tensor_cb_id`) + per-head placement | 1 | `compute/reduction.cpp`, `dataflow/writer_llama_reduce_scatter.cpp` | Portable | pack → output; create-heads split via `get_write_ptr()` + byte offset = L1/NoC addr only | Portable | — |
| `cb_fabric_sender` / `cb_fabric_receiver` (`fabric_sender_cb_id`, `fabric_receiver_cb_id`) | 1 | `dataflow/reader_llama_reduce_scatter.cpp`, `dataflow/writer_llama_reduce_scatter.cpp` | Portable | fabric staging, canonical FIFO | Portable | — |
| `reserved_packet_header_cb_id` / `cb_packet_header` (`packet_header_cb_id`) | 1/6 | reader/writer kernels | Portable | fabric packet-header scratch; `reserve_back` + `get_write_ptr()` | Portable | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)

## Recommended path

Port freely — mechanical `CircularBuffer` → `DataflowBuffer` rename. No field surgery, no runtime API dependency, no LTA prerequisite.
