# CB→DFB Kernel Audit: `plusone`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/plusone/`

**Scope:** `plusone_program_factory.cpp` → kernel: `reader_plusone_interleaved.cpp` (single reader kernel; in-place increment, no writer/compute kernel).

## Overall verdict: GREEN

**Summary:** Single reader kernel operating in place on one input CB via the modern `CircularBuffer` object API. Step-4 litmus scans return **zero** hits — no GATE, no silent-wrong, no ptr surgery, no field reads. Mechanical `CircularBuffer` → `DataflowBuffer` rename only.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_id_in0` | 1 | `reader_plusone_interleaved.cpp` | Portable | in-place read/increment/write of interleaved input via `get_read_ptr()`/`get_write_ptr()` L1 addr; no field access | Portable | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)
