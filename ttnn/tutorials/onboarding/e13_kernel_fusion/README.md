# E13: Kernel Fusion

Fuse collective communication into compute kernels.

## Goal

Learn advanced kernel fusion techniques:
- Fuse CCLs (all-gather, reduce-scatter) into matmul
- Overlap communication with compute
- Design efficient fused kernel patterns

## Reference

- `ttnn/cpp/ttnn/operations/ccl/`
- `ttnn/cpp/ttnn/operations/matmul/device/matmul_op_multi_core_reuse_mcast_*`

## Key Concepts

### Why Fuse CCLs?
- Unfused: Communication blocks compute, extra memory needed
- Fused: Communication overlaps compute, no extra memory

### Overlap Strategies
- **Producer-Consumer**: Reader receives while compute processes
- **Pipelined**: Ping-pong between buffers

### Fused Kernel Design
- Setup for both communication and compute
- Loop receives shard, computes with it, continues
- Write accumulated result at end

## Common Pitfalls

1. **Synchronization complexity** - More edge cases
2. **Buffer sizing** - Must fit both comm and compute buffers
3. **Latency sensitivity** - If comm slow, compute starves
4. **Debug difficulty** - Harder to debug fused kernels
