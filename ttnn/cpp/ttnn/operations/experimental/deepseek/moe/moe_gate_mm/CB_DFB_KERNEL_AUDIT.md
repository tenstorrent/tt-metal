# CB→DFB Kernel Audit: `moe_gate_mm`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/deepseek/moe/moe_gate_mm/`

**Scope:** In-scope device kernels under `device/kernels/` → `compute.cpp`, `dm0.cpp`, `dm1.cpp`, and the SFPU helper headers `bias_bcast_sfpu.h`, `top2_sum_sfpu.h`, `top4_sfpu.h`, `top8_merge_sfpu.h`, `top8_sfpu.h` (pure SFPU/dst math, no CB field access).

## Overall verdict: GREEN

**Summary:** No GATE, no silent-wrong, no ptr surgery, no field reads, no runtime blockers across all kernels. This is a gate matmul followed by top-k selection SFPU: the CBs (`cb_r2c_w` c_0, `cb_s2c_in` c_1, `cb_c2w_rdy` c_2, `cb_w2c_in2..in8` c_3/c_5–c_9, `cb_s2c_out` c_4) are canonical Class 1 linear FIFOs between the DM readers, compute, and writer via `reserve_back`/`push_back`·`wait_front`/`pop_front`. The top-k SFPU headers operate on dest registers only. `get_read_ptr()`/`get_write_ptr()` appear only as L1/NoC byte addresses. Mechanical `CircularBuffer` → `DataflowBuffer` rename.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_r2c_w` (c_0) | 1 | `dm0.cpp`, `dm1.cpp`, `compute.cpp` | Portable | weights reader → compute, linear FIFO | Portable | — |
| `cb_s2c_in` (c_1) | 1 | `dm0.cpp`, `dm1.cpp`, `compute.cpp` | Portable | activation reader → compute, linear FIFO | Portable | — |
| `cb_c2w_rdy` (c_2) | 1 | `dm0.cpp`, `dm1.cpp`, `compute.cpp` | Portable | compute → writer ready/handoff CB, linear FIFO | Portable | — |
| `cb_s2c_out` (c_4) | 1 | `dm0.cpp`, `dm1.cpp`, `compute.cpp` | Portable | gate/top-k output; `get_write_ptr()`/`get_read_ptr()` as L1/NoC addr only | Portable | — |
| `cb_w2c_in2`..`cb_w2c_in8` (c_3, c_5–c_9) | 1 | `dm0.cpp`, `dm1.cpp`, `compute.cpp` | Portable | writer→compute staged top-k operand CBs, linear FIFO (c_6 reused as `in8`) | Portable | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)
