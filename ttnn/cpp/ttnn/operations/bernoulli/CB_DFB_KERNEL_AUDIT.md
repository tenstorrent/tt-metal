# CB→DFB Kernel Audit: `bernoulli`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/bernoulli/`

**Scope:** `bernoulli_program_factory.cpp` → kernels: `reader_bernoulli.cpp`, `writer_bernoulli.cpp`, `compute_bernoulli.cpp`

## Overall verdict: RED

**Summary:** All CBs are Class 1 linear FIFO / private staging scratch. The only blockers are two `get_local_cb_interface(...).fifo_page_size` field reads (reader + writer) — a hard GATE, but a trivial mechanical `get_entry_size()` swap. After the swap this is GREEN.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_in` (c_0) | 1 | `reader_bernoulli.cpp`, `writer_bernoulli.cpp` | Blocked | GATE: `reader_bernoulli.cpp:21` reads `.fifo_page_size` → `get_entry_size()`; otherwise linear FIFO, `get_read_ptr()` cursor | Blocked | same |
| `cb_intermed` (c_24) | 1 | `compute_bernoulli.cpp`, `writer_bernoulli.cpp` | Portable | compute `rand_tile`→`pack`→`push_back`; writer `get_read_ptr()` as `float*` | Portable | — |
| `cb_intermed1` (c_25) | 6 | `writer_bernoulli.cpp` | Blocked | Private staging scratch (`reserve_back` once, no `push_back`, CPU-assembled tile → NOC write) → **ScratchpadSpec** (autoportable). GATE: `writer_bernoulli.cpp:26` reads `.fifo_page_size` → `get_entry_size()` | Blocked | same |

## GATE hits (must be empty to merge)

- `reader_bernoulli.cpp:21` — `get_local_cb_interface(in_cb_id).fifo_page_size` — → `cb_in.get_entry_size()`
- `writer_bernoulli.cpp:26` — `get_local_cb_interface(intermed1_cb_id).fifo_page_size` — → `cb_intermed1.get_entry_size()`

## Blocked on runtime (2xx rollup)

- (none)
