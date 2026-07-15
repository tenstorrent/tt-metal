# CB→DFB Kernel Audit: `deepseek_moe_reduce_scatter`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/ccl/deepseek_moe_reduce_scatter/`

**Scope:** All in-scope device kernels under `device/kernels/`: `deepseek_moe_reduce_scatter_reader.cpp`, `deepseek_moe_reduce_scatter_reduction.cpp`, `deepseek_moe_reduce_scatter_writer.cpp`.

## Overall verdict: GREEN

**Summary:** DeepSeek MoE reduce-scatter = fabric NoC/semaphore movement + a canonical reduction compute kernel. All Step-4 litmus scans return **zero** hits — no GATE, no silent-wrong, no `read_tile_value`/`get_tile_address`, no `get_pointer_to_cb_data`, no ptr surgery, no `fifo_*` field reads. CBs are canonical linear-FIFO input/intermediate slices, a reduced-output CB, and a compute CB. This is a reduce-scatter dataflow op — **not** the out-of-scope `deepseek_moe_gate` / `deepseek_prefill` firmware-reconfig kernels. Mechanical rename.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_input_slice` (`input_slice_cb_id`) | 1 | `deepseek_moe_reduce_scatter_reader.cpp`, `deepseek_moe_reduce_scatter_reduction.cpp` | Portable | input slices, canonical FIFO | Portable | — |
| `cb_intermediate_slice` (`intermediate_slice_cb_id`) | 1 | `deepseek_moe_reduce_scatter_reader.cpp`, `deepseek_moe_reduce_scatter_reduction.cpp` | Portable | fabric-received partials, canonical FIFO | Portable | — |
| `cb_compute` (`compute_cb_id`) | 1 | `deepseek_moe_reduce_scatter_reduction.cpp` | Portable | reduction working buffer, canonical FIFO | Portable | — |
| `cb_reduced` (`reduced_cb_id`) | 1 | `deepseek_moe_reduce_scatter_reduction.cpp`, `deepseek_moe_reduce_scatter_writer.cpp` | Portable | pack → output, `get_write_ptr()` as L1/NoC addr | Portable | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)

## Recommended path

Port freely — mechanical `CircularBuffer` → `DataflowBuffer` rename. No field surgery, no runtime API dependency, no LTA prerequisite.
