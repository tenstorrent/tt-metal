# E08: Multi-Core

Expand the matmul+add kernel to use all tensix cores on one chip.

## Goal

Parallelize the single-core kernel across multiple tensix cores using grid distribution.

## Key Concepts

- Core grid specification
- Work distribution across cores
- Multicast for shared data
- Synchronization primitives

## Reference

- `tt_metal/programming_examples/matmul/matmul_multi_core/` - Multi-core matmul example
- `tt_metal/programming_examples/matmul/matmul_multicore_reuse_mcast/` - Multicast example

## Workflow

1. Modify the kernel to distribute work across a core grid
2. Implement proper synchronization
3. Profile with Tracy to verify parallelization
4. Compare performance against single-core version
