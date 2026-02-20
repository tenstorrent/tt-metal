# E14: Kernel Fusion

Fuse matmul, activation, and collective communication into unified kernels.

## Goal

Learn kernel fusion techniques:
- Fuse activation (from e13) into matmul output
- Fuse CCL operations (from e11) with compute
- Overlap communication with computation
- Eliminate intermediate memory writes

## Prerequisites

- **e11**: Multi-chip, CCLs, reduction
- **e12**: Matmul kernel structure
- **e13**: SFPU activation functions

## Reference

- `ttnn/cpp/ttnn/operations/matmul/device/`
- `ttnn/cpp/ttnn/operations/ccl/`

## Key Concepts

### Why Fuse?

**Matmul + Activation:**
- Unfused: matmul → write → read → activation → write
- Fused: matmul → activation → write (1 write instead of 2)

**Matmul + CCL:**
- Unfused: matmul → write → CCL blocking → next op
- Fused: matmul produces while CCL sends previous output

### Activation Fusion Pattern
```cpp
// In compute kernel, after matmul accumulation:
matmul_tiles(...);       // Result in dst register
gelu_tile(dst_idx);      // Activation in-place
pack_tile(dst_idx, cb);  // Write final result
```

### CCL Fusion Pattern
```cpp
// Overlap communication with compute:
// Reader: receives shard from CCL while compute works
// Compute: processes current shard while next arrives
// Writer: sends result to CCL while compute produces next
```

### Full Fusion (Matmul + Activation + CCL)
- Receive input shard via CCL
- Compute matmul on shard
- Apply activation in-place
- Send result via CCL
- All stages pipelined and overlapped

## Common Pitfalls

1. **Init functions** - Must init matmul, SFPU, and CCL
2. **Synchronization** - More complex with three fused components
3. **Buffer sizing** - Must fit comm, compute, and activation buffers
4. **Latency sensitivity** - Slowest component limits throughput
5. **Debug difficulty** - Isolate components when debugging
