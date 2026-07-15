# CB→DFB Kernel Audit: `all_reduce_create_qkv_heads`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/transformer/all_reduce_create_qkv_heads/`

**Scope:** All in-scope device kernels → `device/kernels/compute/reduction.cpp`, `device/kernels/dataflow/reduction_receiver.cpp`, `device/kernels/dataflow/worker_reader.cpp`, `device/kernels/dataflow/worker_writer.cpp`. Donor includes: CCL `minimal_ccl_common.hpp`, `hetergeneous_data_structs.hpp`, fabric `fabric_connection_manager.hpp`/`noc_addr.h` (all scanned — clean).

## Overall verdict: GREEN

**Summary:** All CBs are canonical Class 1 linear FIFOs or bare `get_read_ptr()`/`get_write_ptr()` L1/NoC byte addresses. Step-4 litmus scans (GATE, silent-wrong, `read_tile_value`/`get_tile_address`, `get_pointer_to_cb_data`, ptr surgery, field reads) return **zero** hits across the kernel closure. The fabric all-reduce staging and the packet-header CB are sync-free scratch/FIFO patterns that port mechanically.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_in0` / `cb_out0` | 1 | `compute/reduction.cpp` | Portable | canonical `wait_front`/`reserve_back`/`pack_tile`/`push_back` accumulate → `DataflowBuffer` | Portable | — |
| `cb0_id` | 1 | `worker_reader.cpp`, `worker_writer.cpp` | Portable | linear FIFO staging for fabric mcast; `get_write_ptr()`/`get_read_ptr()` as L1/NoC addr | Portable | — |
| `reserved_packet_header_cb_id` | 6 | `worker_writer.cpp` | Portable | sync-free packet-header scratch (`reserve_back`/`push_back` + `get_write_ptr()`), no consumer → `ScratchpadSpec` (autoportable) | Portable | same |
| `cb_id` (reduction signal) | 1 | `reduction_receiver.cpp` | Portable | `cb_push_back` credit release after semaphore wait → `DataflowBuffer` | Portable | — |
| `cb_id_reduction_out` | 1/3 | `reduction_receiver.cpp` | Portable (workaround) | **undesirable but OK hack:** `get_read_ptr() + in_tile_offset_by_batch` byte scatter as NoC write source (Class 3 offset read); uplift: strided DFB on Quasar | Portable (workaround) | same |
| `cb_batch_offset_id` | 6 | `reduction_receiver.cpp` | Portable | sync-free index CB; `get_write_ptr()` reinterpret to read batch-offset scalar (not `get_pointer_to_cb_data`) → `ScratchpadSpec`/LTA candidate | Portable | same |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)
