# CB→DFB Kernel Audit: `topk_large_indices`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/topk_large_indices/`

**Scope:** `topk_large_indices_program_factory.cpp` → kernels: `reader.cpp`, `compute.cpp`, `writer.cpp`.

## Overall verdict: GREEN

**Summary:** All CBs are canonical Class 1 FIFOs / staging buffers via the modern `CircularBuffer` object API. Step-4 litmus scans return **zero** hits — no GATE, no silent-wrong, no ptr surgery, no field reads. The `cb_indices_scratch` buffer is a canonical `reserve_back` + `get_write_ptr()` staging region (no ptr/field surgery). Mechanical `CircularBuffer` → `DataflowBuffer` rename only.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_in` / `input_cb` | 1 | `reader.cpp`, `compute.cpp` | Portable | value input tiles, canonical FIFO | Portable | — |
| `indices_cb` / `cb_indices` | 1 | `compute.cpp`, `writer.cpp` | Portable | index tiles, linear FIFO | Portable | — |
| `cb_indices_scratch` | 1/6 | `writer.cpp` | Portable | index write staging via `reserve_back` + `get_write_ptr()` (no field access); optional uplift: `ScratchpadSpec` | Portable | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)
