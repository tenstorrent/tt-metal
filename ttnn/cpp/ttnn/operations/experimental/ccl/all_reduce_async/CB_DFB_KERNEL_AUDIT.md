# CB→DFB Kernel Audit: `all_reduce_async`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/`

**Scope:** All in-scope device kernels under `device/kernels/`: `compute/reduction.cpp`, `dataflow/worker_reader.cpp`, `dataflow/worker_writer.cpp`, `dataflow/reduction_receiver.cpp`.

## Overall verdict: GREEN

**Summary:** Fabric all-reduce = NoC/semaphore movement + a canonical reduction compute kernel. All Step-4 litmus scans return **zero** hits — no GATE, no silent-wrong, no `read_tile_value`/`get_tile_address`, no `get_pointer_to_cb_data`, no ptr surgery, no `fifo_*` field reads. CBs are canonical linear-FIFO inputs/output for the reduction plus a reserved fabric packet-header scratch CB. Mechanical rename.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_in` / `cb_in0` / `cb_in1` (`reduction_input_cb_id`) | 1 | `compute/reduction.cpp`, `dataflow/worker_reader.cpp`, `dataflow/reduction_receiver.cpp` | Portable | reduction inputs, canonical FIFO | Portable | — |
| `cb_out` / `cb_out0` | 1 | `compute/reduction.cpp`, `dataflow/worker_writer.cpp` | Portable | pack → output, `get_write_ptr()` as L1/NoC addr | Portable | — |
| `reserved_packet_header_cb_id` / `cb_packet_header` | 1/6 | `dataflow/worker_reader.cpp`, `dataflow/worker_writer.cpp`, `dataflow/reduction_receiver.cpp` | Portable | fabric packet-header scratch; `reserve_back` + `get_write_ptr()` | Portable | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)

## Recommended path

Port freely — mechanical `CircularBuffer` → `DataflowBuffer` rename. No field surgery, no runtime API dependency, no LTA prerequisite.
