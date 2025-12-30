# Operation Complexity Analysis Tool

This tool analyzes TT-NN operations to rank them by complexity, enabling prioritized refactoring efforts from simplest to most complex cases.

## Overview

The complexity analysis extracts metrics from program factory files (the core files that create `tt::tt_metal::Program` objects) and computes weighted complexity scores. Operations are then bucketed into four categories: Simple, Standard, Complex, and Very Complex.

## Usage

```bash
cd /home/boxx/tt-metal
python3 ttnn/tools/complexity_analysis/analyze_complexity.py
```

This will generate:
- `complexity_report.json` - Detailed metrics in JSON format
- `complexity_ranking.md` - Human-readable ranking with buckets and characteristics

## Complexity Metrics

The tool extracts the following metrics from each program factory file:

### Primary Metrics

- **Lines of Code (LOC)**: Non-empty, non-comment lines
- **CreateKernel calls**: Number of kernels in the program
- **CreateCircularBuffer calls**: Data movement complexity indicator
- **SetRuntimeArgs calls**: Per-core configuration complexity
- **CreateSemaphore calls**: Synchronization complexity

### CCL-Specific Metrics

For CCL (Collective Communication Library) operations, additional metrics capture their unique complexity:

- **create_mesh_workload**: Multi-device coordination calls
- **GlobalSemaphore**: Cross-device synchronization
- **Fabric API calls**: Device-to-device communication (fabric::, mesh_graph, MeshWorkload, MeshDevice)
- **Command stream builders**: CCL command builder usage

### Secondary Metrics

- **Kernel file count**: Total compute + dataflow kernel files
- **Program factory count**: Number of variants/specializations per operation
- **Conditional branches**: if/else/switch statements

## Complexity Score Formula

The weighted complexity score depends on operation type:

### Standard Operations

```
complexity_score = (
    0.3 * normalized_loc +
    0.25 * normalized_kernels +
    0.2 * normalized_cbs +
    0.15 * normalized_runtime_args +
    0.1 * normalized_semaphores
)
```

### CCL Operations

CCL operations use adjusted weights that emphasize mesh workloads and global semaphores:

```
complexity_score = (
    0.25 * normalized_loc +
    0.15 * normalized_kernels +
    0.15 * normalized_cbs +
    0.10 * normalized_runtime_args +
    0.05 * normalized_semaphores +
    0.20 * normalized_mesh_workloads +
    0.10 * normalized_global_semaphores
)
```

All metrics are normalized to 0-1 range before weighting. Operations are automatically detected as CCL if they:
- Have "ccl" in their name, OR
- Use `create_mesh_workload`, OR
- Use `GlobalSemaphore`, OR
- Have significant fabric API usage (>5 calls), OR
- Use command stream builders

## Complexity Buckets

### Simple (0.0 - 0.2)
- Single kernel or no kernels
- Minimal circular buffers (0-2)
- No semaphores
- Low LOC (< 500)
- **Examples**: move, simple data movement operations

### Standard (0.2 - 0.5)
- 2-3 kernels
- Moderate CB usage (3-5)
- Basic runtime args
- Medium LOC (200-1000)
- **Examples**: element-wise operations, simple reductions

### Complex (0.5 - 0.8)
- Multiple program variants
- Sharding support
- 4-7 kernels
- Higher LOC (1000-2000)
- **Examples**: matmul variants, normalization ops

### Very Complex (0.8 - 1.0)
- Multicast operations
- Advanced synchronization (semaphores)
- 8+ kernels
- Very high LOC (2000+)
- Complex runtime arg setup
- **Examples**: matmul_mcast_1d, transformer operations

## Output Files

### complexity_report.json

Structured JSON containing:
- `program_factories`: Per-file metrics for all program factory files
- `operations`: Aggregated metrics and complexity scores per operation

### complexity_ranking.md

Human-readable markdown report with:
- Summary statistics by bucket
- Detailed bucket breakdowns with examples
- Complete ranking (most complex first)
- **CCL Operations Complexity**: Special section highlighting CCL operations with their unique metrics
- **Program Factory Ranking**: All 272 individual program factories ranked by complexity (top 100 shown)
- **Top Programs Per Operation**: For each operation with multiple program factories, shows the most complex variants
- Complexity characteristics for each bucket

## Refactoring Guidance

Use the ranking to:

1. **Start with Simple operations** - Validate refactoring approach on low-complexity cases
2. **Progress to Standard** - Build confidence with moderate complexity
3. **Tackle Complex operations** - Apply proven patterns to challenging cases
4. **Address Very Complex last** - Use accumulated experience for the most difficult refactorings

## Implementation Details

The tool:
1. Enumerates all `*program_factory*.cpp` files under `ttnn/cpp/ttnn/operations/`
2. Extracts metrics using regex pattern matching
3. Aggregates metrics per operation (summing across program factories)
4. Normalizes metrics to 0-1 range
5. Computes weighted complexity scores
6. Buckets operations into four categories
7. Generates JSON and markdown reports

## Notes

- The analysis is based on static code metrics - actual runtime complexity may vary
- Operations with multiple program factories have their metrics summed
- Kernel file counts are approximate (counts all .cpp files in kernels/ directories)
- Operation names are extracted from file paths and may need manual review for accuracy
