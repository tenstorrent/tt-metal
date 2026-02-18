# E12: Matmul Deep Dive

Deep dive into matmul variants and optimization strategies.

## Goal

Understand matmul implementation strategies:
- Compare matmul_1d vs matmul_2d
- Learn blocking strategies
- Understand math fidelity trade-offs

## Reference

- `tt_metal/programming_examples/matmul/`
- `ttnn/cpp/ttnn/operations/matmul/`

## Key Concepts

### Matmul Variants
- **matmul_1d**: Single-core, K-dimension streaming, good for small matrices
- **matmul_2d**: Multi-core with 2D work distribution, good for large matrices

### Blocking Strategies
- Block_M, Block_K, Block_N control tile grouping
- Larger blocks = better reuse but more memory
- Trade-off between parallelism and register pressure

### Math Fidelity
- **HiFi4**: Highest precision, slowest
- **HiFi2**: High precision, medium speed
- **LoFi**: Lower precision, fastest

### Accumulation
- Partial products accumulated across K dimension
- FP32 accumulation for BF16 inputs avoids precision loss

## Common Pitfalls

1. **Register pressure** - Large Block_K exhausts tile registers
2. **L1 overflow** - Large blocks don't fit
3. **Underutilization** - Small matrices don't fill all cores
