# E09: L1 vs DRAM

Understand memory hierarchy and memory placement strategies.

## Goal

Learn the memory hierarchy and when to use each type:
- Understand L1 vs DRAM characteristics
- Learn about memory banking and interleaving
- Optimize memory placement for performance

## Reference

- `docs/source/tt-metalium/tt_metal/advanced_topics/memory_for_kernel_developers.rst`

## Key Concepts

### Memory Hierarchy
- **L1 (SRAM)**: 1.5MB per core, very fast, low latency
- **DRAM**: Several GB total, slower, higher latency

### Memory Banking
- DRAM organized into banks for parallel access
- Interleaved allocation distributes pages across banks
- Avoids bank conflicts for maximum bandwidth

### When to Use Each
- **L1**: Intermediate results, reused data, small tensors
- **DRAM**: Large tensors, input/output, streaming data

## Common Pitfalls

1. **L1 overflow** - Tensors too large cause allocation failures
2. **Bank conflicts** - Same bank access reduces bandwidth
3. **NOC congestion** - Many cores reading same DRAM region
