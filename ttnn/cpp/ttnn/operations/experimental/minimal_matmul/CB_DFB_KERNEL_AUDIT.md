# CB→DFB Kernel Audit: `minimal_matmul`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/minimal_matmul/`

**Scope:** `minimal_matmul_program_factory.cpp` → kernels: `dm_in0_sender.cpp`, `dm_in1_sender_out.cpp`, `compute.cpp`, shared header `matmul_dataflow_common.hpp`; donor include `experimental/ccl/strided_all_gather_async/device/kernels/fused_receiver_utils.hpp`.

## Overall verdict: GREEN

**Summary:** All CBs are canonical Class 1 linear FIFOs via the modern `CircularBuffer` object API. Step-4 litmus scans return **zero** hits across the sender/receiver dataflow kernels, compute, and the CCL `fused_receiver_utils.hpp` donor header — no GATE, no silent-wrong, no ptr surgery, no field reads. Mechanical `CircularBuffer` → `DataflowBuffer` rename only.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_in0`, `cb_in1`, `cb_in2` | 1 | `dm_in0_sender.cpp`, `dm_in1_sender_out.cpp`, `compute.cpp` | Portable | activation/weight/gathered inputs, canonical mcast FIFO | Portable | — |
| `cb_intermediate` | 1 | `compute.cpp` | Portable | matmul partials, linear FIFO (no rd/wr ptr surgery) | Portable | — |
| `cb_bias` | 1 | `compute.cpp` | Portable | bias broadcast, linear FIFO | Portable | — |
| `cb_ternary_a`, `cb_ternary_b` | 1 | `dm_in0_sender.cpp`, `dm_in1_sender_out.cpp`, `compute.cpp` | Portable | fused ternary inputs, canonical FIFO | Portable | — |
| `cb_out` (output) | 1 | `dm_in1_sender_out.cpp`, `compute.cpp` | Portable | pack → output, `get_write_ptr()` as L1/NoC addr only | Portable | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)
