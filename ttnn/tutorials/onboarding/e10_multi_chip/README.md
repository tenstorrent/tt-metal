# E10: Multi-Chip

Split tensor computation across multiple chips and gather results.

## Goal

Distribute matmul+add across 2 chips and implement result gathering.

## Key Concepts

- Device mesh abstraction
- Tensor distribution across chips
- Ethernet-based data transfer
- All-gather operation

## Reference

- `tt_metal/programming_examples/distributed/` - Distributed programming examples

## Workflow

1. Set up a 2-chip device mesh
2. Distribute input tensors across chips
3. Perform parallel computation
4. Gather results back to host
5. Profile with Tracy to analyze inter-chip communication
