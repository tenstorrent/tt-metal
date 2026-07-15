# CB→DFB Kernel Audit: `all_gather_async`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/`

**Scope:** All in-scope device kernels under `device/kernels/`: `minimal_default_reader.cpp`, `minimal_default_writer.cpp`, `broadcast_rm_reader.cpp`, `broadcast_rm_writer.cpp`, `llama_shapes_sharded_reader.cpp`, `llama_shapes_sharded_writer.cpp`.

## Overall verdict: GREEN

**Summary:** Pure fabric NoC/semaphore all-gather movement. All Step-4 litmus scans (GATE, silent-wrong, `read_tile_value`/`get_tile_address`, `get_pointer_to_cb_data`, ptr surgery, `fifo_*` field reads) return **zero** hits. CBs are canonical linear-FIFO input/output plus a reserved fabric packet-header scratch CB; only bare `get_read_ptr()`/`get_write_ptr()` L1/NoC addressing is used. Mechanical `CircularBuffer` → `DataflowBuffer` rename, no prerequisites.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_input` / `cb_output` (input/output page) | 1 | `minimal_default_reader.cpp`, `minimal_default_writer.cpp`, `broadcast_rm_reader.cpp`, `broadcast_rm_writer.cpp`, `llama_shapes_sharded_reader.cpp`, `llama_shapes_sharded_writer.cpp` | Portable | canonical `reserve/push` / `wait/pop`; `get_read_ptr()`/`get_write_ptr()` as L1/NoC addr | Portable | — |
| `reserved_packet_header_cb_id` / `cb_packet_header` | 1/6 | all reader/writer kernels | Portable | fabric packet-header scratch; `reserve_back` + `get_write_ptr()` fill, no field access | Portable | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)

## Recommended path

Port freely — mechanical `CircularBuffer` → `DataflowBuffer` rename. No field surgery, no runtime API dependency, no LTA prerequisite.
