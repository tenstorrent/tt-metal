# Tilize-Untilize Multi-Core Implementation Plan

## Overview

This document outlines the implementation plan for adding multi-core support to the `tilize_untilize` operation. The plan is based on the patterns established in the reference `tilize_multi_core_interleaved` and `untilize_multi_core` implementations.

**Current State**: Single-core implementation working at `tilize_untilize_program_factory.cpp`
**Target**: Multi-core interleaved implementation for improved performance on larger tensors

## Work Distribution Strategy

### Reference Pattern Analysis

Both `tilize` and `untilize` multi-core interleaved variants use the `split_blocks_for_tilize()` utility from `work_split_tilize.hpp`:

```cpp
auto [ncores, all_cores, core_range, core_range_cliff, nblocks_per_core, nblocks_per_core_cliff] =
    ttnn::split_blocks_for_tilize(grid, nblocks);
```

**Key Insight**: This distributes tile rows (blocks) across cores. Each block consists of:
- **Height**: 32 rows (one tile height)
- **Width**: Full tensor width (`num_tiles_per_row` tiles)

### Work Unit Definition

For tilize_untilize:
- **Work unit**: One tile row = 32 input sticks producing `num_tiles_per_row` output sticks
- **Total work**: `nblocks = num_tile_rows * batch_size` (same as current single-core)
- **Distribution**: Each core processes `nblocks_per_core` consecutive tile rows

### Core Distribution

```
nblocks = (tensor_height / TILE_HEIGHT) * batch_size

split_blocks_for_tilize() produces:
- ncores: Number of cores to use
- core_range: Full cores processing nblocks_per_core each
- core_range_cliff: Single cliff core processing remainder (nblocks_per_core_cliff)
```

**Example**: 8 tile rows on 3 cores
- Cores 0, 1: 3 blocks each (nblocks_per_core = 3)
- Core 2: 2 blocks (nblocks_per_core_cliff = 2)

## Design Decisions

### Decision 1: Use Height-Based 1D Parallelization

**Choice**: Distribute work along the height (tile row) dimension only.

**Rationale**:
- Matches the established tilize/untilize multi-core interleaved pattern
- Simpler than 2D block distribution (`split_blocks_for_tilize_wh`)
- Each core processes complete tile rows, maintaining data locality
- Reader reads consecutive sticks, writer writes consecutive sticks

**Alternatives Considered**:
- 2D block distribution: More complex, used only for very wide tensors (>32 tiles per row)
- Column parallelization: Only beneficial for single-tile-height wide tensors

**When to use 2D block**: The reference tilize implementation switches to block mode when:
```cpp
if (num_tiles_per_row > 32) {
    if (num_tiles_per_col > 32 || num_tiles_per_row > num_tiles_per_col) {
        // Use 2D block distribution
    }
}
```
This is out of scope for initial multi-core implementation.

### Decision 2: Runtime Block Count (Single Kernel for All Cores)

**Choice**: Move `num_blocks` from compile-time to runtime argument for compute kernel.

**Rationale**:
- Different cores process different numbers of blocks (full cores vs cliff core)
- A single kernel binary works for all cores when block count is runtime
- Simpler than creating separate kernel handles for cliff cores
- The loop `for (block = 0; block < num_blocks; ++block)` naturally handles varying block counts

**Current (single-core)**:
```cpp
// Compile-time args for compute
std::vector<uint32_t> compute_compile_time_args = {num_blocks, num_tiles_per_row};
```

**Proposed (multi-core)**:
```cpp
// Compile-time: only tiles_per_row (constant across cores)
std::vector<uint32_t> compute_compile_time_args = {num_tiles_per_row};

// Runtime: blocks per core (varies per core)
// Full cores get nblocks_per_core, cliff core gets nblocks_per_core_cliff
SetRuntimeArgs(program, compute_kernel_id, core, {blocks_for_this_core});
```

**Why reference implementations use separate kernels**: The tilize/untilize reference implementations use compile-time block counts, forcing them to create separate kernel handles for cliff cores. By using runtime args, we avoid this complexity.

**Cliff core handling**: With runtime block count, the cliff core simply receives `nblocks_per_core_cliff` as its runtime argument - no separate kernel needed.

### Decision 3: Circular Buffer Sizing (Double-Buffering)

**Choice**: Use double-buffered CBs when core processes 2+ blocks.

**Rationale**:
- Enables overlap between reader-compute and compute-writer
- Follows the untilize_multi_core pattern
- Single-buffering is fine for cores processing 1 block

**Sizing Logic**:
```cpp
uint32_t cb_num_tiles;
if (nblocks_per_core == 1) {
    cb_num_tiles = num_tiles_per_row;  // Single buffer
} else {
    cb_num_tiles = num_tiles_per_row * 2;  // Double buffer
}
```

**Trade-off**: Uses more L1 memory per core but enables pipelining.

## Implementation Plan

### Step 1: Program Factory Changes

**File**: `tilize_untilize_program_factory.cpp`

1. Add work split calculation:
```cpp
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"

// Get available cores
auto grid_size = device->compute_with_storage_grid_size();

// Split work across cores
auto [ncores, all_cores, core_range, core_range_cliff, nblocks_per_core, nblocks_per_core_cliff] =
    ttnn::split_blocks_for_tilize(grid_size, num_blocks);
```

2. Update CB sizing for double-buffering:
```cpp
uint32_t cb_tiles = (nblocks_per_core > 1) ? num_tiles_per_row * 2 : num_tiles_per_row;

// CB_in, CB_tiled, CB_out all use this sizing
```

3. Create kernels on `all_cores` CoreRangeSet instead of single core:
```cpp
// Single kernel for all cores (full + cliff) - runtime block count handles the difference
KernelHandle compute_kernel_id = CreateKernel(
    program,
    "ttnn/cpp/ttnn/operations/reduction/tilize_untilize/device/kernels/compute/tilize_untilize_compute.cpp",
    all_cores,
    ComputeConfig{.fp32_dest_acc_en = fp32_dest_acc_en, .compile_args = {num_tiles_per_row}});
```

4. Set per-core runtime arguments (unified loop for all cores):
```cpp
uint32_t row_start_id = 0;
auto cores = corerange_to_cores(all_cores);
bool has_cliff = !core_range_cliff.empty();

for (uint32_t i = 0; i < ncores; ++i) {
    const CoreCoord& core = cores[i];

    // Determine block count for this core
    bool is_cliff_core = has_cliff && (i == ncores - 1);
    uint32_t blocks_for_this_core = is_cliff_core ? nblocks_per_core_cliff : nblocks_per_core;

    // Reader: src_addr, num_sticks, start_stick_id
    SetRuntimeArgs(program, reader_kernel_id, core, {
        src_buffer->address(),
        blocks_for_this_core * TILE_HEIGHT,  // num_sticks
        row_start_id
    });

    // Writer: dst_addr, num_blocks, start_stick_id
    SetRuntimeArgs(program, writer_kernel_id, core, {
        dst_buffer->address(),
        blocks_for_this_core,
        row_start_id
    });

    // Compute: num_blocks (runtime) - same kernel handles all cores
    SetRuntimeArgs(program, compute_kernel_id, core, {blocks_for_this_core});

    row_start_id += TILE_HEIGHT * blocks_for_this_core;
}
```

5. Update shared variables to store all cores info:
```cpp
struct TilizeUntilizeSharedVariables {
    KernelHandle reader_kernel_id;
    KernelHandle compute_kernel_id;
    KernelHandle writer_kernel_id;
    CoreRangeSet all_cores;
    std::vector<CoreCoord> cores;
    uint32_t num_cores;
};
```

### Step 2: Reader Kernel Changes

**File**: `reader_tilize_untilize_interleaved.cpp`

Current behavior is already multi-core ready:
- Uses `start_stick_id` to offset into tensor
- Processes `num_sticks` rows
- No changes needed

### Step 3: Writer Kernel Changes

**File**: `writer_tilize_untilize_interleaved.cpp`

Current behavior is already multi-core ready:
- Uses `start_stick_id` to offset output
- Processes `num_blocks` blocks
- No changes needed

### Step 4: Compute Kernel Changes

**File**: `tilize_untilize_compute.cpp`

**Change**: Move `num_blocks` from compile-time to runtime argument.

**Before**:
```cpp
constexpr uint32_t num_blocks = get_compile_time_arg_val(0);
constexpr uint32_t num_tiles_per_row = get_compile_time_arg_val(1);
```

**After**:
```cpp
constexpr uint32_t num_tiles_per_row = get_compile_time_arg_val(0);
const uint32_t num_blocks = get_arg_val<uint32_t>(0);  // Runtime
```

### Step 5: Update Shared Variables and Override

Update `override_runtime_arguments()` to handle all cores:
```cpp
void TilizeUntilizeProgramFactory::override_runtime_arguments(...) {
    auto& cores = shared_vars.cores;
    for (const auto& core : cores) {
        auto& reader_args = GetRuntimeArgs(program, reader_kernel_id, core);
        reader_args[0] = src_buffer->address();

        auto& writer_args = GetRuntimeArgs(program, writer_kernel_id, core);
        writer_args[0] = dst_buffer->address();
    }
}
```

## Runtime Argument Summary

### Reader Kernel (per-core)
| Index | Name | Description |
|-------|------|-------------|
| 0 | src_addr | Source buffer DRAM address |
| 1 | num_sticks | Sticks to read (nblocks * 32) |
| 2 | start_stick_id | Starting row index |

### Writer Kernel (per-core)
| Index | Name | Description |
|-------|------|-------------|
| 0 | dst_addr | Destination buffer DRAM address |
| 1 | num_blocks | Blocks to write |
| 2 | start_stick_id | Starting output row index |

### Compute Kernel (per-core)
| Index | Name | Description |
|-------|------|-------------|
| 0 | num_blocks | Blocks to process |

## Performance Considerations

### Expected Benefits
1. **Parallelism**: Work distributed across N cores (up to num_tile_rows cores)
2. **Bandwidth**: Multiple NOC reads/writes in parallel
3. **Pipelining**: Double-buffering enables overlap within each core

### Potential Bottlenecks
1. **Small tensors**: May not have enough work to justify multi-core overhead
2. **Memory bandwidth**: DRAM bandwidth may become bottleneck with many cores
3. **L1 pressure**: Double-buffering increases per-core L1 usage

### When Single-Core May Be Better
- Very small tensors (< 4 tile rows)
- When debugging
- When other operations need the cores

## Open Questions

### Q1: Should we add a mode selection flag?

**Options**:
1. Always use multi-core (simplest)
2. Automatic selection based on tensor size
3. User-configurable via operation attributes

**Recommendation**: Start with automatic selection:
```cpp
bool use_multicore = (nblocks >= MIN_BLOCKS_FOR_MULTICORE);  // e.g., 4
```

### Q2: Should cliff core get different CB sizing?

The cliff core processes fewer blocks, potentially needing less CB space.

**Options**:
1. Same sizing for all cores (simpler, current approach)
2. Different sizing for cliff (more L1 efficient)

**Recommendation**: Same sizing - simplicity outweighs small L1 savings.

### Q3: How to handle the intermediate CB_tiled?

The intermediate CB (c_1) holds tiled data between tilize and untilize. With double-buffering:
- Should it be double-buffered? Currently single-buffered.
- The compute kernel processes one block at a time, so single-buffering may be sufficient.

**Recommendation**: Keep CB_tiled single-buffered. The tilize->untilize happens atomically per block, so no overlap is possible within that path.

### Q4: Should 2D block distribution be added?

For very wide tensors, 2D block distribution (`split_blocks_for_tilize_wh`) can use more cores.

**Recommendation**: Out of scope for initial implementation. Add later if performance profiling shows it's needed for specific use cases.

## Testing Strategy

### Unit Tests
1. Single tile (32x32) - minimum case
2. Multiple tiles per row (32x128)
3. Multiple tile rows (128x128)
4. Large tensor (512x512)
5. Non-square tensor (32x1024, 1024x32)
6. Batch dimension (2x2x64x64)

### Verification
- Output matches single-core implementation exactly (bit-exact)
- All cores receive correct work assignments
- No hangs or deadlocks
- Cliff core handling works correctly

### Performance Tests
- Measure speedup vs single-core for various tensor sizes
- Profile NOC bandwidth utilization
- Measure L1 usage per core

## Implementation Checklist

- [ ] Add work split calculation to program factory
- [ ] Update CB sizing for double-buffering
- [ ] Create single kernel on all_cores CoreRangeSet (unified for full + cliff cores)
- [ ] Update compute kernel to use runtime num_blocks
- [ ] Implement per-core runtime argument setting (unified loop handles cliff)
- [ ] Update shared variables structure
- [ ] Update override_runtime_arguments
- [ ] Add unit tests
- [ ] Performance benchmarking

## References

### Code References
- `ttnn/cpp/ttnn/operations/core/work_split/work_split_tilize.hpp` - Work distribution utilities
- `ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_multi_core_interleaved_program_factory.cpp` - Tilize multi-core pattern
- `ttnn/cpp/ttnn/operations/data_movement/untilize/device/factories/untilize_multi_core_program_factory.cpp` - Untilize multi-core pattern

### Analysis Documents
- `tilize_analysis.md` - Tilize operation analysis
- `untilize_analysis.md` - Untilize operation analysis
- `tilize_untilize_spec.md` - Operation specification
