# CB→DFB Kernel Audit: `generic`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/generic/`

**Scope:** `generic_op_program_factory.cpp` (`GenericMeshDescriptorFactory`) — no in-scope device kernels

## Overall verdict: GREEN (N/A)

**Summary:** The generic op is a pass-through that returns a caller-supplied `ProgramDescriptor` (`operation_attributes.mesh_programs`). It defines **no kernels of its own** under `generic/` — there is nothing to scan or gate here. CB→DFB portability is entirely a property of whatever kernels the caller supplies, which are outside this op's scope and must be audited at their source op.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| (none in-scope) | — | — | — | Kernels/CBs are provided by the caller's `ProgramDescriptor` | — | audit at the donor op |

## GATE hits (must be empty to merge)

- (none — no in-scope kernels)

## Blocked on runtime (2xx rollup)

- (none)
