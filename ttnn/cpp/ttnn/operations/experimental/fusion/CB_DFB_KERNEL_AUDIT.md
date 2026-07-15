# CB→DFB Kernel Audit: `fusion` (fusion_dispatch_op)

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/fusion/`

**Scope:** `fusion_dispatch_op_device_operation` + `fusion_dispatch_op_helpers.hpp`. No `device/kernels/` directory exists and no kernel-source path literals are assigned in the op's device code.

## Overall verdict: GREEN (no in-scope device kernels)

**Summary:** `fusion_dispatch_op` is a **descriptor-dispatch** op: it consumes pre-built `KernelDescriptor`s (assembled from other ops via the Python entry points in `fusion_dispatch_op_nanobind.cpp`) and only patches their runtime args / buffer addresses on host — it ships **no device kernels of its own**. Scope discovery finds zero kernel sources under `fusion/`, so there is **nothing to scan or gate** here. CB→DFB port readiness for the fused sub-ops is owned by each donor op's own audit (the descriptors carry whatever CBs/DFBs the source op defined). There are no CBs, no GATE hits, and no runtime blockers in this op root.

## Scope notes

- The `desc.kernels[...]` references in `fusion_dispatch_op_helpers.hpp` index into caller-supplied descriptor structs (host bookkeeping), not kernel-source paths — no device code to audit.
- This is **not** an OUT-OF-SCOPE (MOE-gate) case; it simply has no kernel closure. Track the fused sub-ops under their own `CB_DFB_KERNEL_AUDIT.md` files.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| (none) | — | — | — | no device kernels in this op root | — | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)
