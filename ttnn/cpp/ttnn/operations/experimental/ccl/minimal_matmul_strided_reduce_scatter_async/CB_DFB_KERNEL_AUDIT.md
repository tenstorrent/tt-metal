# CB→DFB Kernel Audit: `minimal_matmul_strided_reduce_scatter_async`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/ccl/minimal_matmul_strided_reduce_scatter_async/`

**Scope:** No kernels of its own — pure host wrapper. `device/minimal_matmul_strided_reduce_scatter_async_program.cpp` composes two donor program factories and creates no `device/kernels/`:
- `strided_reduce_scatter_async` — `build_ring_strided_reduce_scatter_async_program_artifacts` (donor: `ttnn/cpp/ttnn/operations/experimental/ccl/strided_reduce_scatter_async/`).
- `minimal_matmul` — `minimal_matmul_factory_helper_common` (donor: `ttnn/cpp/ttnn/operations/experimental/minimal_matmul/`).

## Overall verdict: GREEN (inherited)

**Summary:** This op has **no device kernels** — it delegates to the `strided_reduce_scatter_async` and `minimal_matmul` program factories. Verdict inherits from the donors, both of which litmus-scan **clean** (zero GATE / silent-wrong / `read_tile_value` / `get_pointer_to_cb_data` / ptr-surgery / `fifo_*` field-read hits): `strided_reduce_scatter_async` is canonical fabric reduce-scatter dataflow, and `minimal_matmul` is a self-contained canonical matmul (the same clean family as `all_gather_minimal_matmul_async`, **not** the RED `bmm_*_gathered` compute). Combined rollup: **GREEN**.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| (all CBs) | — | delegated — no own kernels | — | inherits `strided_reduce_scatter_async` (GREEN) + `minimal_matmul` (GREEN) — canonical FIFO + bare-pointer L1/NoC addressing | — | same |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)

## Recommended path

No kernel work in this op. Both donors port mechanically (`CircularBuffer` → `DataflowBuffer` rename) with no field surgery, runtime API dependency, or LTA prerequisite. This wrapper inherits GREEN.
