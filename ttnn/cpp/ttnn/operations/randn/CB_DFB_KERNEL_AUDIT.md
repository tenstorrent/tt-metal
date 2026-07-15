# CB→DFB Kernel Audit: `randn`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/randn/`

**Scope:** `randn_program_factory.cpp` → kernels: `compute_standard_normal.cpp`, `writer_standard_normal.cpp`

## Overall verdict: GREEN

**Summary:** Clean, modern kernel (2025). Single Class 1 linear FIFO, uses only sanctioned APIs (`get_tile_size()`, `get_cb_id()`, `get_read_ptr()` as L1 cursor). No `get_local_cb_interface` field access, no `read_tile_value`. Mechanical `CircularBuffer` → `DataflowBuffer` rename only.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_dst` (c_0) | 1 | `compute_standard_normal.cpp`, `writer_standard_normal.cpp` | Portable | Box-Muller compute packs 1–2 tiles → writer `wait_front(2)`/`pop_front`, `get_tile_size()` + `get_read_ptr()` NOC source | Portable | canonical PACK→UNPACK single-ended packer → optional **SELF-LOOP-CANDIDATE** |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)
