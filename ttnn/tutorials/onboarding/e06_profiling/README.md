# E06: Profiling

Calculate theoretical peak performance and measure actual performance with Tracy.

## Goal

Learn to profile and analyze kernel performance:
- Calculate theoretical peak based on hardware specs
- Capture and analyze Tracy profiles
- Identify compute-bound vs memory-bound kernels

## Reference

- `docs/source/tt-metalium/tools/tracy_profiler.rst`
- `docs/source/tt-metalium/tools/device_program_profiler.rst`

## Key Concepts

### Peak Performance Calculation
- Compute peak: cores × ops_per_cycle × clock_frequency
- Memory bandwidth peak: DRAM channels × channel_bandwidth
- Compare actual vs theoretical to find bottleneck

### Tracy Profiling
- Enable with `TT_METAL_DEVICE_PROFILER=1`
- Run with `python -m tracy {test_script}.py`
- Captures detailed execution traces
- Shows kernel timings, NOC transactions, stalls

### Identifying Bottlenecks
- **Compute-bound**: Math units busy, DRAM idle
- **Memory-bound**: Waiting for data, compute underutilized

## Common Pitfalls

1. **Profiling overhead** - Tracy adds overhead; don't trust absolute times
2. **Cold vs warm runs** - First run includes compilation
3. **Small workloads** - Launch overhead dominates for small tensors
