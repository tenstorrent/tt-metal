# E15: Pipelining

Master double buffering and latency hiding techniques.

## Goal

Learn advanced pipelining techniques:
- Implement double buffering for circular buffers
- Overlap reader/compute/writer stages
- Maximize throughput by hiding latency

## Reference

- `tt_metal/programming_examples/matmul/`

## Key Concepts

### Double Buffering
- Use two buffer slots to overlap operations
- While computing on buffer A, read into buffer B
- Swap and repeat

### Circular Buffer Configuration
- Single buffering: num_tiles = 1 (blocking)
- Double buffering: num_tiles = 2 (overlapped)
- Triple buffering: num_tiles = 3 (more overlap)

### Pipelining Pattern
- Reader prefetches next tile while compute uses current
- Compute processes while reader fetches
- Writer writes while compute produces next

### Three-Stage Pipeline
- Reader, Compute, Writer operating concurrently
- Requires CB depth of at least 2 between stages

## Common Pitfalls

1. **CB depth too small** - Pipeline stalls
2. **Unbalanced stages** - Slowest stage limits throughput
3. **Pipeline drain** - Don't forget last tiles
4. **Memory pressure** - Double buffering uses 2x memory
