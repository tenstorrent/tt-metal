# CB→DFB Kernel Audit: `llama_reduce_scatter`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter/`

**Scope:** All in-scope device kernels under `device/kernels/`: `compute/reduction.cpp`, `dataflow/reader_llama_reduce_scatter.cpp`, `dataflow/writer_llama_reduce_scatter.cpp`.

## Overall verdict: GREEN

**Summary:** Llama reduce-scatter = fabric NoC/semaphore movement + a canonical reduction/accumulator compute kernel. All Step-4 litmus scans return **zero** hits — no GATE, no silent-wrong, no `read_tile_value`/`get_tile_address`, no `get_pointer_to_cb_data`, no ptr surgery, no `fifo_*` field reads. CBs are canonical linear-FIFO input/output/accumulator, fabric sender/receiver staging CBs, and a reserved packet-header scratch CB. The accumulator uses canonical FIFO (no rd/wr ptr save/restore). Mechanical rename.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_input_tensor` (`input_tensor_cb_id`) | 1 | `dataflow/reader_llama_reduce_scatter.cpp`, `compute/reduction.cpp` | Portable | input tiles, canonical FIFO | Portable | — |
| `cb_accumulator` (`accumulator_cb_id`) | 1 | `compute/reduction.cpp` | Portable | reduction accumulator via canonical `reserve/push` / `wait/pop` — **no** rd/wr ptr surgery | Portable | — |
| `cb_output_tensor` (`output_tensor_cb_id`) | 1 | `compute/reduction.cpp`, `dataflow/writer_llama_reduce_scatter.cpp` | Portable | pack → output, `get_write_ptr()` as L1/NoC addr | Portable | — |
| `cb_fabric_sender` / `cb_fabric_receiver` (`fabric_sender_cb_id`, `fabric_receiver_cb_id`) | 1 | `dataflow/reader_llama_reduce_scatter.cpp`, `dataflow/writer_llama_reduce_scatter.cpp` | Portable | fabric staging, canonical FIFO + `get_read_ptr()`/`get_write_ptr()` | Portable | — |
| `reserved_packet_header_cb_id` / `cb_packet_header` (`packet_header_cb_id`) | 1/6 | reader/writer kernels | Portable | fabric packet-header scratch; `reserve_back` + `get_write_ptr()` | Portable | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)

## Recommended path

Port freely — mechanical `CircularBuffer` → `DataflowBuffer` rename. No field surgery, no runtime API dependency, no LTA prerequisite.
