# E11: Kernel Fusion

Fuse all_gather into the matmul_add kernel for better multi-chip performance.

## Goal

Eliminate separate all_gather operation by fusing it into the compute kernel.

## Key Concepts

- Kernel fusion benefits (reduced memory traffic, lower latency)
- Overlapping communication and compute
- Fused kernel design patterns

## Workflow

1. Profile the multi-chip solution to identify communication overhead
2. Design a fused kernel that combines gather + compute
3. Implement the fused kernel
4. Profile and compare against unfused version
5. Document speedup and trade-offs
