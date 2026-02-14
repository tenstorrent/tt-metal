# E07: L1 vs DRAM

Understand memory hierarchy and compare L1 vs DRAM performance.

## Goal

Use L1 memory for intermediate compute results and compare performance against DRAM-only approach.

## Key Concepts

- L1: Fast on-chip SRAM per tensix core (~1MB)
- DRAM: Larger but slower off-chip memory
- Memory bandwidth bottlenecks
- When to use L1 vs DRAM

## Reference

- `docs/source/tt-metalium/tt_metal/advanced_topics/memory_for_kernel_developers.rst` - Memory guide for kernel developers

## Workflow

1. Modify the kernel to use L1 for intermediate results
2. Profile with Tracy
3. Compare against DRAM-based version
4. Document bandwidth and latency differences
