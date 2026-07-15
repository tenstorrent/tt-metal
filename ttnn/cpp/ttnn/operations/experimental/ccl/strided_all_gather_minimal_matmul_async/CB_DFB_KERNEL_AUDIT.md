# CB→DFB Kernel Audit: `experimental/ccl/strided_all_gather_minimal_matmul_async`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/ccl/strided_all_gather_minimal_matmul_async/`

**Scope:** Host wrapper — **no kernels of its own** under `device/kernels/`.

## Overall verdict: GREEN (delegation)

**Summary:** This is a **fused composite host op with no device kernels of its own**. `StridedAllGatherMinimalMatmulAsyncProgramFactory` (`device/strided_all_gather_minimal_matmul_async_program.cpp`) builds one program by delegating to two other factories:

1. `StridedAllGatherAsyncProgramFactory::strided_all_gather_async_minimal_default_helper(...)` — donor kernels from **`experimental/ccl/strided_all_gather_async`** (`minimal_default_reader.cpp`, `minimal_default_writer.cpp`, `strided_all_gather_common.hpp`).
2. `ttnn::experimental::prim::MinimalMatmulProgramFactory` — donor kernels from **`experimental/minimal_matmul`**.

Because it references no kernel paths of its own, there is nothing to scan directly; its CB→DFB readiness is the **rollup of its two donors**. The Step-4 litmus scans on both donor kernel sets return **zero hits** (GATE, silent-wrong, QUASAR-BLOCKED, LTA, ptr-surgery, field-reads all clear), so the composite is **GREEN**.

## CB portability

| Donor op | Verdict | Kernels | Notes |
|----------|---------|---------|-------|
| `strided_all_gather_async` | GREEN | `minimal_default_reader.cpp`, `minimal_default_writer.cpp`, `strided_all_gather_common.hpp` | `cb_output` canonical Class 1 FIFO — see `strided_all_gather_async/CB_DFB_KERNEL_AUDIT.md` |
| `minimal_matmul` (`experimental/minimal_matmul`) | GREEN | matmul reader/writer/compute kernels | zero litmus hits — canonical Class 1 matmul FIFOs |

## GATE hits (must be empty to merge)

- (none) — no owned kernels; both donor kernel sets are GATE-clean.

## Blocked on runtime (2xx rollup)

- (none)

## Recommended path

No kernel work in this directory. Port the two donor ops (`strided_all_gather_async` and `minimal_matmul`) — both are mechanical `CircularBuffer` → `DataflowBuffer` renames with no field surgery, runtime API, or LTA prerequisite. When both donors are ported, this composite inherits the port with no additional device-side change.
