# CB→DFB Kernel Audit: `test/hang_device`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/test/hang_device/`

**Scope:** `hang_device_program_factory.cpp` → kernel: `device/kernels/compute/hang_device_kernel.cpp` (single compute kernel; test-only op that intentionally hangs the device).

## Overall verdict: GREEN

**Summary:** Test-only op. The single compute kernel declares **no CBs** and does no dataflow — it is a deliberate hang used for device/timeout testing. Step-4 litmus scans return **zero** hits. Nothing to port (no CB→DFB surface); trivially GREEN.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| (none) | — | `hang_device_kernel.cpp` | Portable | test kernel, no CB usage | Portable | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)
