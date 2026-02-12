# E12: Pipelining

Overlap compute and data movement for maximum throughput.

## Goal

Implement double buffering to hide memory latency by overlapping data fetch with computation.

## Key Concepts

- Double buffering: Use two buffers alternately for fetch and compute
- Pipeline stages: Reader, Compute, Writer operating concurrently
- Hiding memory latency
- Buffer management and synchronization

## Workflow

1. Profile the fused kernel to identify stalls
2. Implement double buffering in circular buffer usage
3. Ensure proper synchronization between stages
4. Profile with Tracy to verify overlap
5. Measure and document throughput improvement
