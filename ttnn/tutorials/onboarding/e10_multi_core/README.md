# E10: Multi-Core

Expand kernels to use multiple Tensix cores.

## Goal

Learn to parallelize across cores:
- Distribute work across a core grid
- Use multicast for shared data
- Synchronize with semaphores

## Reference

- `tt_metal/programming_examples/matmul/matmul_multi_core/`
- `tt_metal/programming_examples/matmul/matmul_multicore_reuse_mcast/`

## Key Concepts

### Core Grid Specification
- Define which cores run your kernel with `CoreRange` / `CoreRangeSet`
- Can be 1D (row of cores) or 2D (grid)

### Work Distribution
- Split work evenly across cores
- Handle edge cases when work doesn't divide evenly
- Each core computes its portion based on core ID

### Multicast
- Broadcast data from one core to many
- Useful for shared weights in matmul
- More efficient than each core reading separately

### Semaphores
- Synchronize between cores
- Wait for signal before proceeding
- Coordinate producer-consumer patterns

## Common Pitfalls

1. **Uneven work** - Last core may have less work
2. **Mcast deadlock** - Receivers must be ready before sender
3. **Core ID confusion** - Logical vs physical coordinates
