# CB→DFB Kernel Audit: `topk_router_gpt`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/topk_router_gpt/`

**Scope:** `topk_router_gpt_program_factory.cpp` → kernels: `dm0.cpp`, `dm1.cpp`, `compute.cpp`.

## Overall verdict: GREEN

**Summary:** All CBs are canonical Class 1 linear FIFOs via the modern `CircularBuffer` object API with heavy but standard `reserve_back`/`push_back`/`wait_front`/`pop_front` usage across the two dataflow kernels and compute. Step-4 litmus scans return **zero** hits — no GATE, no silent-wrong, no ptr surgery, no field reads, no runtime-blocked APIs. Mechanical `CircularBuffer` → `DataflowBuffer` rename only.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_weight`, `cb_input`, `cb_bias` | 1 | `dm0.cpp`, `compute.cpp` | Portable | matmul inputs, canonical FIFO | Portable | — |
| `cb_partial_recv`, `cb_local_out` | 1 | `dm1.cpp`, `compute.cpp` | Portable | partial gather / local output, linear FIFO | Portable | — |
| `cb_index`, `cb_topk_val` | 1 | `dm1.cpp`, `compute.cpp` | Portable | topk index/value tiles | Portable | — |
| `cb_gathered_val`, `cb_gathered_ind` | 1 | `dm1.cpp`, `compute.cpp` | Portable | gathered values/indices, canonical FIFO | Portable | — |
| `cb_intermed_val`, `cb_intermed_ind` | 1 | `compute.cpp` | Portable | topk intermediates, linear FIFO | Portable | — |
| `cb_softmax_mask`, `cb_softmax_tmp`, `cb_reduce_scalar`, `cb_bcast_scaler` | 1 | `dm1.cpp`, `compute.cpp` | Portable | softmax mask/scratch/scalars, canonical FIFO | Portable | — |
| `cb_final_out`, `cb_dispatch` | 1 | `dm1.cpp`, `compute.cpp` | Portable | final output / dispatch tiles, pack → output | Portable | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)
