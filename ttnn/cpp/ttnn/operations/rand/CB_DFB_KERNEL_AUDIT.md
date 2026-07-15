# CB→DFB Kernel Audit: `rand`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/rand/`

**Scope:** `rand_program_factory.cpp` → **donor kernels (uniform):** `uniform/device/kernels/writer_uniform.cpp`, `uniform/device/kernels/compute_uniform.cpp`

## Overall verdict: RED

**Summary:** `rand` has **no kernels of its own** — it reuses uniform's `writer_uniform.cpp` / `compute_uniform.cpp` (donor). Classification is identical to `uniform`: Class 1 + private staging, blocked only by the shared `fifo_page_size` GATE in the donor writer. Fixing it in uniform clears both ops.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_intermed` (c_24) | 1 | `compute_uniform.cpp` (donor), `writer_uniform.cpp` (donor) | Portable | compute produce → writer consume, linear FIFO | Portable | — |
| `cb_dst` (c_0) | 6 | `writer_uniform.cpp` (donor) | Blocked | conversion staging → **ScratchpadSpec**; GATE: donor `writer_uniform.cpp:26` `.fifo_page_size` → `get_entry_size()` | Blocked | same |

## GATE hits (must be empty to merge)

- `uniform/device/kernels/writer_uniform.cpp:26` (donor) — `get_local_cb_interface(dst_cb_id).fifo_page_size` — → `get_entry_size()`

## Blocked on runtime (2xx rollup)

- (none)
