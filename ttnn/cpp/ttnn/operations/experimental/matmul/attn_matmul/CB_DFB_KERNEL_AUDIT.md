# CB→DFB Kernel Audit: `attn_matmul`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/matmul/attn_matmul/`

**Scope:** `attn_matmul_program_factory.cpp` → kernels: `dataflow/reader_transformer_attn_matmul.cpp`, `compute/transformer_attn_matmul.cpp`; **donor** writer `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`.

## Overall verdict: YELLOW

**Summary:** The in-tree reader and compute kernels are clean canonical Class 1 FIFOs (zero litmus hits). The only finding is in the **cross-op donor writer** `writer_unary_interleaved_start_id.cpp:19`, which reads `get_local_cb_interface(cb_id_out).fifo_page_size`. That is a **mechanical NEEDS-FIX** — swap to `get_entry_size()` (getter exists today, no runtime dependency) — and clears the GATE with a one-line change. All CBs are otherwise mechanically portable.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_in0`, `cb_in1` | 1 | `reader_transformer_attn_matmul.cpp`, `transformer_attn_matmul.cpp` | Portable | activation/weight inputs, canonical FIFO | Portable | — |
| `cb_intermed0`, `cb_intermed1`, `cb_intermed2` | 1 | `reader_transformer_attn_matmul.cpp`, `transformer_attn_matmul.cpp` | Portable | transpose/matmul intermediates, linear FIFO | Portable | — |
| `cb_out` (output) | 1 | `transformer_attn_matmul.cpp`, `writer_unary_interleaved_start_id.cpp` (donor) | Portable | **NEEDS-FIX:** donor writer `get_local_cb_interface(cb_id_out).fifo_page_size` (`writer_unary_interleaved_start_id.cpp:19`) → `get_entry_size()` (getter exists) | Portable | same |

## GATE hits (must be empty to merge)

- `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp:19` — `get_local_cb_interface(cb_id_out).fifo_page_size` **read** — **mechanical**: → `get_entry_size()` (getter exists today). Shared donor writer (also used by other unary/matmul ops); clears with a trivial swap.

## Blocked on runtime (2xx rollup)

- (none)
