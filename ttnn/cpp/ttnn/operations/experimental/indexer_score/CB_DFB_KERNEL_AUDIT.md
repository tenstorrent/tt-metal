# CB→DFB Kernel Audit: `indexer_score`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/indexer_score/`

**Scope:** `indexer_score_program_factory.cpp` → kernels: `reader_indexer_score.cpp`, `writer_indexer_score.cpp`, `compute_indexer_score.cpp`, headers `indexer_score_common.hpp`, `indexer_score_cb.hpp`, `indexer_score_work_split.hpp`; donor include `transformer/sdpa/device/kernels/dataflow/dataflow_common.hpp`.

## Overall verdict: GREEN

**Summary:** All CBs are canonical Class 1 linear FIFOs driven through the modern `CircularBuffer` object API (`reserve_back`/`push_back`/`wait_front`/`pop_front`). The Step-4 litmus scans return **zero** hits across the kernel closure (including the sdpa `dataflow_common.hpp` donor) — no GATE, no silent-wrong, no ptr surgery, no field reads, no runtime-blocked APIs. Mechanical `CircularBuffer` → `DataflowBuffer` rename only.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_q`, `cb_k`, `cb_w`, `cb_mask` | 1 | `reader_indexer_score.cpp`, `compute_indexer_score.cpp` | Portable | mcast/interleaved inputs, canonical FIFO | Portable | — |
| `cb_qk`, `cb_acc_strip`, `cb_out_strip` | 1 | `compute_indexer_score.cpp`, `writer_indexer_score.cpp` | Portable | matmul/accumulate pipeline, linear FIFO | Portable | — |
| `cb_scaler`, `cb_pool_scratch` | 1 | `compute_indexer_score.cpp` | Portable | reduce scalar + pool scratch, `reserve/push` staging (autoportable: `ScratchpadSpec` if refactored, not required) | Portable | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)
