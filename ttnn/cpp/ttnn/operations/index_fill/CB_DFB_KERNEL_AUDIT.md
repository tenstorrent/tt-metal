# CB→DFB Kernel Audit: `index_fill`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/index_fill/`

**Scope:** `index_fill_multi_core_factory.cpp` → kernels: `reader_index_fill.cpp`, `writer_index_fill.cpp`

## Overall verdict: GREEN

**Summary:** No field access, no `read_tile_value`. Two Class 1 FIFOs, one private pre-fill scratch, and the documented WEIRD-OK in-place CB edit (guide's Class 6 flagship for index_fill). All portable now.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_src` (c_0) | 1 / 6 | `reader_index_fill.cpp`, `writer_index_fill.cpp` | Portable (workaround) | **undesirable but OK hack:** in-place edit `input_ptr[col]=fill_value` on `wait_front`'d tile via `get_read_ptr()`, then NOC write (documented WEIRD-OK) | Portable (workaround) | same; optional harden via `CoreLocalMem` + `scoped_lock` |
| `cb_index` (c_1) | 1 | `reader_index_fill.cpp`, `writer_index_fill.cpp` | Portable | whole index tensor `push_back` once → `wait_front`, `get_read_ptr()` as `uint32_t*` lookup | Portable | — |
| `cb_fill` (c_2) | 6 | `writer_index_fill.cpp` | Portable | writer-local pre-fill scratch (`get_write_ptr()` fill, NOC-write source, no consumer) → **ScratchpadSpec** (autoportable) | Portable | same |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)
