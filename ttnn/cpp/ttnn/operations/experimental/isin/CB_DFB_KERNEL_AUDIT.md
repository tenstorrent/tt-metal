# CB→DFB Kernel Audit: `isin`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/isin/`

**Scope:** `isin_program_factory.cpp` → kernels: `dataflow/isin_reader.cpp`, `dataflow/isin_writer.cpp`, shared header `dataflow/isin_common.hpp`.

## Overall verdict: GREEN

**Summary:** Both dataflow kernels route every CB through the modern `CircularBuffer` object API with canonical `reserve_back`/`push_back` (producer) and `wait_front`/`pop_front` (consumer) single-page handoffs. Step-4 litmus scans return **zero** hits — no GATE, no silent-wrong, no ptr surgery, no field reads. Mechanical `CircularBuffer` → `DataflowBuffer` rename only.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| input / test-element / output CBs | 1 | `isin_reader.cpp`, `isin_writer.cpp`, `isin_common.hpp` | Portable | canonical one-page FIFO handoff (`push_page`/`pop_page` helpers wrap `reserve/push` + `wait/pop`) | Portable | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)
