# CB→DFB Kernel Audit: `matmul_wo`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/deepseek/mla/matmul_wo/`

**Scope:** In-scope device kernels under `device/kernels/` → `compute.cpp`, `compute_collector.cpp`, `dm0.cpp`, `dm1.cpp`, `dm1_collector.cpp`, `matmul_wo_ring_common.h` (compile-time ring layout constants only, no CB access).

## Overall verdict: GREEN

**Summary:** No GATE, no silent-wrong, no ptr surgery, no field reads, no runtime blockers across all five kernels. This is a ring matmul over 12 cores: the CBs (`cb_r2c_w` c_0, `cb_s2c_in` c_1, `cb_c2w_out` c_2, `cb_s2c_in2` c_3, `cb_s2c_out` c_4) are canonical Class 1 linear FIFOs handing tiles between the DM readers, the compute engines, and the collector kernels via `reserve_back`/`push_back`·`wait_front`/`pop_front`. `get_read_ptr()`/`get_write_ptr()` appear only as L1/NoC byte addresses. Mechanical `CircularBuffer` → `DataflowBuffer` rename.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_r2c_w` (c_0) | 1 | `dm0.cpp`, `dm1.cpp`, `compute.cpp` | Portable | weights reader → compute, linear FIFO | Portable | — |
| `cb_s2c_in` (c_1) | 1 | `dm0.cpp`, `dm1.cpp`, `compute.cpp` | Portable | activation reader → compute, linear FIFO | Portable | — |
| `cb_c2w_out` (c_2) | 1 | `dm0.cpp`, `dm1.cpp`, `compute.cpp` | Portable | compute → writer output; `get_write_ptr()` as L1/NoC addr only | Portable | — |
| `cb_s2c_in2` (c_3) | 1 | `dm1.cpp`, `dm1_collector.cpp`, `compute_collector.cpp` | Portable | ring-collector input, linear FIFO | Portable | — |
| `cb_s2c_out` (c_4) | 1 | `dm1_collector.cpp`, `compute_collector.cpp` | Portable | collector output, linear FIFO | Portable | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)
