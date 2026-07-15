# CB→DFB Kernel Audit: `uniform`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/uniform/`

**Scope:** `uniform_program_factory.cpp` → kernels: `writer_uniform.cpp`, `compute_uniform.cpp`

## Overall verdict: RED

**Summary:** Class 1 FIFO + one private conversion-staging CB. Single blocker: `get_local_cb_interface(dst_cb_id).fifo_page_size` read in the writer — mechanical `get_entry_size()` swap → GREEN. (These same kernels are reused as donors by `rand`.)

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_intermed` (c_24) | 1 | `compute_uniform.cpp`, `writer_uniform.cpp` | Portable | compute `rand_tile`→`pack`→`push_back`; writer `wait_front`/`pop_front`, `get_read_ptr()` cursor (fp32 NOC-writes directly) | Portable | — |
| `cb_dst` (c_0) | 6 | `writer_uniform.cpp` | Blocked | bf16 conversion staging (`reserve_back` once, `push_back` once, never streamed) → **ScratchpadSpec**. GATE: `writer_uniform.cpp:26` `.fifo_page_size` → `get_entry_size()` | Blocked | same |

## GATE hits (must be empty to merge)

- `writer_uniform.cpp:26` — `get_local_cb_interface(dst_cb_id).fifo_page_size` — → `cb_dst.get_entry_size()`

## Blocked on runtime (2xx rollup)

- (none)
