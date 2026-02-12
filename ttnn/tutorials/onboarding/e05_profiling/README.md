# E05: Profiling

Debug with DPRINT and analyze performance with Tracy profiler.

## Goal

Learn to debug and profile kernels:
- Use DPRINT to debug kernel execution
- Use Tracy to analyze performance
- Identify bottlenecks (compute vs memory bound)

## Key Concepts

- **DPRINT debugging**: TT_METAL_DPRINT_CORES environment variable
- **Core types**: UNPACK, MATH, PACK cores
- **Tracy profiling**: Launching capture, interpreting timelines
- **Bottleneck analysis**: Compute vs memory bound identification
- **NOC transaction analysis**: Understanding data movement costs

## Reference

- `docs/source/tt-metalium/tools/tracy_profiler.rst` - Tracy setup and usage guide

## Workflow

1. Run a kernel with Tracy profiling enabled
2. Analyze the trace to understand kernel execution
3. Document observations about performance characteristics
