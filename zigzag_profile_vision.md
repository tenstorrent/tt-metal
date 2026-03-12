# Feature Context: Single-Device Profiling for Ring Joint SDPA

## Long-Term Goal

**Isolate and profile the pure compute performance of ring_joint_sdpa in causal+balanced mode**, removing inter-device communication overhead from measurements.

This enables:
1. Understanding the true compute cost per device
2. Identifying compute bottlenecks without noise from fabric/CCL overhead
3. Comparing theoretical vs actual compute efficiency
4. Optimizing the compute kernels independently of communication

---

## The Problem

When profiling `ring_joint_sdpa` on a multi-device system:
- **What we measure**: Total time = Compute + AllGather + Synchronization + Fabric latency
- **What we want**: Pure compute time for one device's workload

The ring all-gather fuses with SDPA compute, making it impossible to separate compute time from communication time in production runs.

---

## The Solution

Create a **single-device profiling op** (`ring_joint_sdpa_profile`) that:

1. **Reuses the exact compute kernel** - No approximations, same code path
2. **Replaces all-gather with DRAM reads** - Pre-stage "gathered" KV in DRAM in the exact order it would arrive during ring iterations
3. **Removes synchronization overhead** - No semaphore waits, no fabric signaling
4. **Simulates one device's view** - Given `ring_index` and `ring_size`, computes exactly what that device would compute

---

## Journey / Implementation Phases

### Phase 1: Correctness Test (Current Step)
**Goal**: Verify the profiling op produces correct results

- Write minimal test: `ring_size=2`, `ring_index=0`, causal+balanced, no joint tensors
- Implement helper functions for chunk ordering and data preparation
- Create PyTorch reference that computes expected output
- Validate profiling op output matches reference

**Success Criteria**: Test passes with PCC > 0.99

### Phase 2: Full Op Implementation
**Goal**: Implement the profiling op with all kernel changes

- Create new device operation files
- Copy and modify reader kernel (remove gather, add DRAM reads)
- Copy and modify writer kernel (remove sync)
- Reuse compute kernel exactly
- Create program factory (simplified, no CCL setup)
- Add Python bindings

**Success Criteria**: Profiling op compiles and runs on single device

### Phase 3: Extended Test Coverage
**Goal**: Validate across configurations

- Test all devices in ring (`ring_index` = 0, 1, ..., ring_size-1)
- Test different ring sizes (2, 4, 8)
- Test with joint tensors
- Test various sequence lengths and chunk sizes
- Compare against multi-device runs (output should match)

**Success Criteria**: All configurations pass correctness checks

### Phase 4: Profiling Integration
**Goal**: Enable performance measurement

- Integrate with Tracy/device profiler
- Measure kernel duration breakdown:
  - Reader time (DRAM → L1)
  - Compute time (QK matmul, softmax, PV matmul)
  - Writer time (L1 → DRAM)
- Compare against theoretical peak

**Success Criteria**: Can generate performance reports per kernel

### Phase 5: Optimization Insights
**Goal**: Use profiling data to optimize

- Identify if compute-bound or memory-bound
- Profile different `q_chunk_size` / `k_chunk_size` configurations
- Compare streaming vs non-streaming compute paths
- Find optimal configurations per sequence length

**Success Criteria**: Actionable optimization recommendations

### Phase 6: Automated Benchmarking (Optional)
**Goal**: Continuous performance tracking

- Add profiling op to benchmark suite
- Track compute efficiency over time
- Detect performance regressions in compute kernels

---

## Key Design Decisions

### Why reuse compute kernel exactly?
- Ensures profiling measures the actual production code path
- Any optimization found applies directly to production
- No risk of profiling "optimized-for-benchmarks" code

### Why pre-stage KV in DRAM in arrival order?
- Matches the exact memory access pattern of production
- Reader kernel indexing logic stays the same
- DRAM bandwidth is still part of the measurement (realistic)

### Why simulate a single device, not the whole ring?
- One device's compute is representative (balanced workload)
- Simpler to implement and debug
- Can run on single-device systems
- Avoids needing multi-device setup for profiling

### Why include local KV in gathered buffer (Option A)?
- Simplifies kernel indexing (same offset calculation)
- First ring iteration reads from offset 0
- Consistent with production kernel's view of memory

---

## What This Does NOT Measure

- **Fabric bandwidth** - No actual inter-device transfers
- **AllGather latency** - No ring communication
- **Synchronization overhead** - No semaphore waits
- **Multi-device contention** - Single device isolation

These are intentionally excluded to isolate compute performance.

---

## Expected Insights

1. **Compute efficiency**: How close to theoretical peak matmul throughput?
2. **Memory boundedness**: Is DRAM bandwidth the bottleneck?
3. **Chunk size sensitivity**: How does performance vary with q/k chunk sizes?
4. **Balanced vs unbalanced**: Does balanced mode have compute overhead?
5. **Scaling behavior**: How does per-device compute scale with sequence length?

---

## Related Work

- `ring_distributed_sdpa` - Similar zigzag pattern, different use case (causal-only, no joint)
- `joint_sdpa` - Single-device joint attention (no ring)
- Standard `sdpa` - Single-device causal/non-causal attention

The profiling op bridges `ring_joint_sdpa` and single-device profiling capabilities.

---

## File References

- **Plan**: `local.md` - Detailed implementation plan
- **Context**: `context.md` - Codebase details and kernel analysis
- **Test location**: `tests/ttnn/unit_tests/operations/sdpa/test_ring_joint_sdpa_profile.py` (to be created)
