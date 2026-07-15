# CB→DFB Kernel Audit: `gelu_backward`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/unary_backward/gelu_backward/`

**Scope:** `gelu_backward_program_factory.cpp` → compute kernels `device/kernels/compute/eltwise_bw_gelu_approx_tanh.cpp`, `device/kernels/compute/eltwise_bw_gelu_poly.cpp`; **donor** reader `eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp`; **donor** writer `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`.

## Overall verdict: YELLOW

**Summary:** The in-tree compute kernels (both the `tanh`-approx and polynomial branches) are clean canonical Class 1 FIFOs (zero litmus hits), and the binary reader donor is clean. The only finding is in the **cross-op donor writer** `writer_unary_interleaved_start_id.cpp:19`, which reads `get_local_cb_interface(cb_id_out).fifo_page_size`. That is a **mechanical NEEDS-FIX** — swap to `get_entry_size()` (getter exists today) — and clears the GATE with a one-line change. All CBs are otherwise mechanically portable.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_grad_out`, `cb_input` | 1 | `reader_binary_interleaved_start_id.cpp` (donor), `eltwise_bw_gelu_approx_tanh.cpp`, `eltwise_bw_gelu_poly.cpp` | Portable | grad-output + input tiles, canonical FIFO | Portable | — |
| `cb_grad_in` (output) | 1 | `eltwise_bw_gelu_*.cpp`, `writer_unary_interleaved_start_id.cpp` (donor) | Portable | **NEEDS-FIX:** donor writer `get_local_cb_interface(cb_id_out).fifo_page_size` (`writer_unary_interleaved_start_id.cpp:19`) → `get_entry_size()` (getter exists) | Portable | same |

## GATE hits (must be empty to merge)

- `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp:19` — `get_local_cb_interface(cb_id_out).fifo_page_size` **read** — **mechanical**: → `get_entry_size()` (getter exists today). Shared donor writer (also used by other unary/matmul ops); clears with a trivial swap.

## Blocked on runtime (2xx rollup)

- (none)
