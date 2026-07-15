# CB→DFB Kernel Audit: `deepseek_moe_post_combine_tilize`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/deepseek_moe_post_combine_tilize/`

**Scope:** In-scope device kernels under `device/kernels/` → `deepseek_moe_post_combine_tilize_compute.cpp`, `deepseek_moe_post_combine_tilize_reader.cpp`, `deepseek_moe_post_combine_tilize_writer.cpp`.

## Overall verdict: GREEN

**Summary:** No GATE, no silent-wrong, no ptr surgery, no field reads, no runtime blockers. A simple tilize pipeline: the reader fills `cb_tilize_input`, the compute `fast_tilize_block`s it into `cb_tilize_output`, and the writer drains it — both CBs are canonical Class 1 linear FIFOs (`reserve_back`/`push_back`·`wait_front`/`pop_front`). `get_read_ptr()`/`get_write_ptr()` are used only as L1/NoC byte addresses. Mechanical `CircularBuffer` → `DataflowBuffer` rename.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_tilize_input` | 1 | `deepseek_moe_post_combine_tilize_reader.cpp`, `deepseek_moe_post_combine_tilize_compute.cpp` | Portable | row-major input read → compute; canonical `reserve/push`·`wait/pop` | Portable | — |
| `cb_tilize_output` | 1 | `deepseek_moe_post_combine_tilize_compute.cpp`, `deepseek_moe_post_combine_tilize_writer.cpp` | Portable | tilized output → writer; `get_read_ptr()` as L1/NoC addr only | Portable | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)
