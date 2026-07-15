# CB→DFB Kernel Audit: `multi_scale_deformable_attn`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/multi_scale_deformable_attn/`

**Scope:** `multi_scale_deformable_attn_program_factory.cpp` → kernels: `dataflow/reader_msda.cpp`, `dataflow/writer_msda.cpp`, `compute/msda_compute.cpp`, shared header `msda_tile_layout.hpp`.

## Overall verdict: GREEN

**Summary:** All CBs are canonical Class 1 FIFOs / staging buffers via the modern `CircularBuffer` object API with `reserve_back`/`push_back`/`wait_front`/`pop_front`. Step-4 litmus scans return **zero** hits — no GATE, no silent-wrong, no ptr surgery, no field reads. The `*_scratch_cb` buffers are used as canonical `reserve_back` + `get_write_ptr()` staging regions (no ptr surgery, no field access); they port mechanically and could optionally be refactored to `ScratchpadSpec` later (not required).

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `input_tile_cb` / `input_cb`, `grid_cb`, `attn_cb`, `scalar_tile_cb` / `scalar_cb` | 1 | `reader_msda.cpp`, `msda_compute.cpp` | Portable | value/grid/attn/scalar inputs, canonical FIFO | Portable | — |
| `value_scratch_cb` | 1/6 | `reader_msda.cpp` | Portable | value gather staging via `reserve_back` + `get_write_ptr()` offsets (no field access) | Portable | — |
| `output_cb`, `output_tile_cb` | 1 | `msda_compute.cpp`, `writer_msda.cpp` | Portable | pack → output, linear FIFO | Portable | — |
| `output_scratch_cb` | 1/6 | `writer_msda.cpp` | Portable | output staging via `reserve_back` + `get_write_ptr()` (no field access) | Portable | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)
