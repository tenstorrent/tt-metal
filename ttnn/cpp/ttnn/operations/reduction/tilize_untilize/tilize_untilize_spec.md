# Tilize-Untilize Template Operation Functional Specification

## Overview
- **Operation Name**: `tilize_untilize`
- **Category**: data_movement (template for compute operations)
- **Reference Operations**: `tilize`, `untilize`, `permute/transpose_xw_tiled`
- **Reference Analyses**:
  - `ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_analysis.md`
  - `ttnn/cpp/ttnn/operations/data_movement/untilize/device/untilize_analysis.md`

## Purpose

This operation serves as a **template for compute operations** that need to:
1. Accept row-major interleaved input data
2. Convert to tiled format for hardware-native computation
3. Perform compute operations on tiled data (placeholder in template)
4. Convert back to row-major format
5. Write row-major interleaved output data

The template establishes the data flow pattern and circular buffer structure for operations like reductions, elementwise transforms, and other compute-intensive workloads that benefit from tiled data processing.

## Mathematical Definition

### Formula
```
output[n, c, h, w] = f(input[n, c, h, w])

where f() is the identity function in this template:
output[n, c, h, w] = input[n, c, h, w]
```

### Semantic Description
The template operation performs an identity transformation: data flows from row-major input through tilization, an empty compute stage (placeholder for future operations), untilization, and back to row-major output. The output is identical to the input.

This template is designed to be extended by replacing the placeholder compute stage with actual math operations (e.g., reductions, activations, normalizations) that operate on tiled data using the matrix engine.

### Data Layout Transformation Flow
```
Row-Major Input (NCHW)          Tiled Intermediate           Row-Major Output (NCHW)
[row0: e0, e1, ..., e31]   →    Tile(0,0): 32x32 block  →   [row0: e0, e1, ..., e31]
[row1: e0, e1, ..., e31]        (with 16x16 faces)          [row1: e0, e1, ..., e31]
...                              ...                         ...
[row31: e0, e1, ..., e31]                                   [row31: e0, e1, ..., e31]
```

## API Specification

### Parameters
| Parameter | Type | Required | Valid Range | Default | Description |
|-----------|------|----------|-------------|---------|-------------|
| input_tensor | Tensor | Yes | - | - | Input tensor in row-major layout |
| output_memory_config | MemoryConfig | No | DRAM/L1 | DRAM_MEMORY_CONFIG | Memory configuration for output |
| output_dtype | DataType | No | Same as input | input.dtype() | Output data type |

### Input Tensor Requirements
| Property | Requirement | Error Message Hint |
|----------|-------------|-------------------|
| Rank | Must be 4D | "Input tensor must be 4D (NCHW)" |
| Layout | ROW_MAJOR_LAYOUT | "Input must be in ROW_MAJOR layout" |
| Memory | INTERLEAVED | "Input must be interleaved (not sharded)" |
| Device | Must be on device | "Input must be on device" |
| Dtype | BFLOAT16 or FLOAT32 | "Unsupported dtype" |
| Height | Multiple of 32 | "Height must be multiple of TILE_HEIGHT (32)" |
| Width | Multiple of 32 | "Width must be multiple of TILE_WIDTH (32)" |

### Output Tensor Specification
| Property | Value/Calculation |
|----------|-------------------|
| Shape | Same as input shape |
| Dtype | Same as input (or specified output_dtype) |
| Layout | ROW_MAJOR_LAYOUT |
| Memory | INTERLEAVED |

## Comparison with Reference Operations

### What Can Be Reused from Tilize
| Aspect | Tilize Reference | This Operation |
|--------|------------------|----------------|
| Reader kernel pattern | reader_unary_stick_layout_split_rows_interleaved | Directly reused |
| Row-major reading | 32 rows at a time (one tile height) | Same pattern |
| CB_in configuration | Single-buffered for tile block | Same |
| Work unit definition | Block of tiles (32 rows × num_tiles_per_row) | Same |

### What Can Be Reused from Untilize
| Aspect | Untilize Reference | This Operation |
|--------|-------------------|----------------|
| Writer kernel pattern | writer_unary_stick_layout_split_rows_multi_core | Directly reused |
| Row-major writing | 32 rows at a time from untilized CB | Same pattern |
| CB_out configuration | Matched to input block size | Same |
| pack_untilize usage | Hardware-accelerated untilization | Same |

### What Can Be Reused from transpose_xw_tiled
| Aspect | Transpose Reference | This Operation |
|--------|---------------------|----------------|
| Three-CB pattern | cb_in → cb_tilize → cb_out | Same structure |
| tilize_helpers usage | compute_kernel_lib::tilize() | Same |
| pack_untilize_dest | Untilize from DEST/CB | Similar pattern |
| Compute kernel structure | tilize → op → untilize | Same (op = identity) |

### Key Differences
| Aspect | Reference Ops | This Operation | Implementation Impact |
|--------|---------------|----------------|----------------------|
| Compute operation | Transpose (wh_tile) | Identity (placeholder) | Simpler compute, extensible |
| Intermediate CB | Used for transpose result | Used for tiled staging | Same CB structure |
| Data flow | Layout transform | Round-trip transform | Same CB count needed |
| Purpose | Data movement only | Template for compute | Extensibility focus |

## Design Decisions

### Decision 1: Three Circular Buffer Architecture
- **Choice**: Use three CBs - CB_in (c_0), CB_tiled (c_1), CB_out (c_16)
- **Rationale**:
  - CB_in holds row-major input ready for tilization
  - CB_tiled holds intermediate tiled data for compute operations
  - CB_out holds row-major output ready for writer
  - This matches the proven pattern from transpose_xw_tiled
- **Alternatives Considered**:
  - Two CBs with in-place transform: Not possible due to format change
  - Single CB with ping-pong: More complex synchronization
- **Tradeoffs**: Uses more L1 memory but provides clean separation and extensibility

### Decision 2: Single-Core Implementation for Template
- **Choice**: Start with single-core, multi-core ready structure
- **Rationale**:
  - Simpler to validate correctness
  - Template focuses on data flow pattern, not parallelization
  - Multi-core extension is straightforward using existing split_blocks_for_tilize()
- **Alternatives Considered**:
  - Multi-core from start: More complex, harder to debug
- **Tradeoffs**: Lower performance for large tensors, but clearer template

### Decision 3: Use Helper Libraries
- **Choice**: Use compute_kernel_lib::tilize() and compute_kernel_lib::untilize()
- **Rationale**:
  - Well-tested, optimized implementations
  - Handles hardware initialization and cleanup
  - Template-based compile-time optimization
- **Alternatives Considered**:
  - Raw LLK calls: More error-prone, harder to maintain
- **Tradeoffs**: Slightly less control, but much cleaner code

### Decision 4: Block-Based Processing
- **Choice**: Process one tile block (32 rows × width_in_tiles) at a time
- **Rationale**:
  - Matches tilize/untilize work unit granularity
  - Efficient CB utilization
  - Natural fit for tile-based hardware
- **Alternatives Considered**:
  - Tile-by-tile: More overhead, poorer cache behavior
  - Full tensor: Exceeds L1 capacity for large tensors
- **Tradeoffs**: Good balance of efficiency and memory usage

### Decision 5: Tile-Aligned Constraint
- **Choice**: Require height and width to be multiples of 32
- **Rationale**:
  - Simplifies implementation significantly
  - No padding logic needed
  - User specified this constraint
  - Template can be extended for non-aligned cases later
- **Alternatives Considered**:
  - Support arbitrary dimensions: Requires tilize_with_val_padding complexity
- **Tradeoffs**: Less flexible but much simpler

## Work Distribution

### Work Unit Definition
One **tile block** consisting of:
- **Height**: 32 rows (TILE_HEIGHT)
- **Width**: num_tiles_per_row × 32 columns

### Single-Core Strategy (Template Default)
- All work processed on core (0, 0)
- Iterate through tensor height in tile-block increments
- Process full width per iteration

### Multi-Core Extension (Future)
```
Work split using split_blocks_for_tilize():
- ncores = min(available_cores, num_tile_rows)
- Each core processes nblocks_per_core tile rows
- Last core may have fewer blocks (cliff handling)
```

### Parallelization Ready Structure
- Work indices (start_block, end_block) passed as runtime args
- Per-core tile addressing via start_tile_id
- Reader/writer kernel structure supports multi-core

## Data Flow

### High-Level Flow
```
DRAM (row-major)
      |
      v
[Reader Kernel / RISCV_0 / NOC0]
- Reads 32 sticks (rows) at a time
- Each stick = tensor_width × element_size bytes
- Blocks of num_tiles_per_row tiles
      |
      v
CB_in (c_0) - Row-major input staging
      |
      v
[Compute Kernel / TENSIX]
- Phase 1: tilize_block(CB_in → CB_tiled)
- Phase 2: [PLACEHOLDER FOR MATH OPERATIONS]
- Phase 3: untilize_block(CB_tiled → CB_out)
      |
      v
CB_out (c_16) - Row-major output staging
      |
      v
[Writer Kernel / RISCV_1 / NOC1]
- Writes 32 rows at a time
- Each row to correct output stick position
      |
      v
DRAM (row-major)
```

### Kernel Data Movement
| Kernel | Core | NOC | Actual Function |
|--------|------|-----|-----------------|
| reader_tilize_untilize | RISCV_0 (BRISC) | NOC0 | Reads row-major sticks from DRAM input |
| compute_tilize_untilize | TENSIX (TRISC) | - | Tilize → [compute placeholder] → Untilize |
| writer_tilize_untilize | RISCV_1 (NCRISC) | NOC1 | Writes row-major sticks to DRAM output |

### Detailed Compute Kernel Flow
```cpp
// Phase 1: Tilize - convert row-major to tiled format
compute_kernel_lib::tilize(CB_in, num_tiles_per_row, CB_tiled, 1);

// Phase 2: Math Operations (PLACEHOLDER)
// Future operations would go here, working on tiled data in CB_tiled
// Example: reduction, activation, normalization, etc.
// For template: data passes through unchanged

// Phase 3: Untilize - convert tiled back to row-major
cb_wait_front(CB_tiled, num_tiles_per_row);
cb_reserve_back(CB_out, num_tiles_per_row);
compute_kernel_lib::untilize<num_tiles_per_row, CB_tiled, CB_out>(1);
cb_pop_front(CB_tiled, num_tiles_per_row);
```

### Circular Buffer Requirements
| CB ID | Name | Purpose | Producer | Consumer | Sizing Strategy |
|-------|------|---------|----------|----------|-----------------|
| c_0 | CB_in | Row-major input staging | Reader (RISCV_0) | Compute (tilize) | num_tiles_per_row tiles × input_tile_size |
| c_1 | CB_tiled | Tiled intermediate | Compute (tilize) | Compute (untilize) | num_tiles_per_row tiles × tile_size |
| c_16 | CB_out | Row-major output staging | Compute (untilize) | Writer (RISCV_1) | num_tiles_per_row tiles × output_tile_size |

### CB Sizing Details
```cpp
// Input tile size for row-major staging (32 rows × 32 cols × element_size)
uint32_t input_tile_size = TILE_HW * element_size;  // 1024 * 2 = 2048 for BF16

// Standard tile size for tiled format
uint32_t tile_size = tt::tile_size(data_format);  // 2048 for BF16, 4096 for FP32

// CB capacities (single-buffered for simplicity)
uint32_t cb_in_size = num_tiles_per_row * input_tile_size;
uint32_t cb_tiled_size = num_tiles_per_row * tile_size;
uint32_t cb_out_size = num_tiles_per_row * tile_size;  // untilized output same size
```

## Memory Access Patterns

### Reader Kernel (RISCV_0 / NOC0) Access
**Function**: Read 32 consecutive rows of input tensor per tile block

```cpp
// For each tile block (32 rows):
for (uint32_t block = 0; block < num_blocks; block++) {
    // Get base addresses for 32 consecutive rows
    for (uint32_t row = 0; row < TILE_HEIGHT; row++) {
        base_src_noc_addr[row] = get_noc_addr(stick_id++, tensor_accessor);
    }

    // Read full width for these 32 rows
    cb_reserve_back(CB_in, num_tiles_per_row);
    for (uint32_t row = 0; row < TILE_HEIGHT; row++) {
        noc_async_read(base_src_noc_addr[row], l1_write_addr, row_width_bytes);
        l1_write_addr += row_width_bytes;
    }
    noc_async_read_barrier();
    cb_push_back(CB_in, num_tiles_per_row);
}
```

**Access Pattern**: Sequential row reads, one stick per NOC transaction, barrier per block

### Writer Kernel (RISCV_1 / NOC1) Access
**Function**: Write 32 consecutive rows of untilized output per tile block

```cpp
// For each tile block (32 rows):
for (uint32_t block = 0; block < num_blocks; block++) {
    cb_wait_front(CB_out, num_tiles_per_row);
    uint32_t l1_read_addr = get_read_ptr(CB_out);

    // Write 32 rows to output
    for (uint32_t row = 0; row < TILE_HEIGHT; row++) {
        uint64_t dst_noc_addr = get_noc_addr(output_stick_id++, tensor_accessor);
        noc_async_write(l1_read_addr, dst_noc_addr, row_width_bytes);
        l1_read_addr += row_width_bytes;
    }
    noc_async_write_barrier();
    cb_pop_front(CB_out, num_tiles_per_row);
}
```

**Access Pattern**: Sequential row writes, one stick per NOC transaction, barrier per block

### Compute Access Pattern
```cpp
// Tilize: wait for row-major input, produce tiled output
cb_wait_front(CB_in, num_tiles_per_row);
cb_reserve_back(CB_tiled, num_tiles_per_row);
tilize_block(CB_in, num_tiles_per_row, CB_tiled);
cb_push_back(CB_tiled, num_tiles_per_row);
cb_pop_front(CB_in, num_tiles_per_row);

// [PLACEHOLDER: Math operations on tiled data in CB_tiled]

// Untilize: wait for tiled input, produce row-major output
cb_wait_front(CB_tiled, num_tiles_per_row);
cb_reserve_back(CB_out, num_tiles_per_row);
untilize_block(CB_tiled, num_tiles_per_row, CB_out);
// OR use pack_untilize for hardware acceleration
cb_push_back(CB_out, num_tiles_per_row);
cb_pop_front(CB_tiled, num_tiles_per_row);
```

## Compile-Time Arguments

### Reader Kernel: `reader_tilize_untilize.cpp`
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | stick_size | uint32_t | Size of one input row in bytes |
| 1+ | TensorAccessorArgs | - | Source buffer access parameters |

### Compute Kernel: `compute_tilize_untilize.cpp`
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_blocks_per_core | uint32_t | Number of tile blocks to process |
| 1 | num_tiles_per_row | uint32_t | Width in tiles (tiles per block) |

### Writer Kernel: `writer_tilize_untilize.cpp`
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_out | uint32_t | Output circular buffer ID (16) |
| 1 | output_stick_size | uint32_t | Size of one output row in bytes |
| 2 | tile_height | uint32_t | Tile height constant (32) |
| 3+ | TensorAccessorArgs | - | Destination buffer access parameters |

## Runtime Arguments

### Reader Kernel
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer DRAM address |
| 1 | num_sticks | uint32_t | Total number of rows to read |
| 2 | start_stick_id | uint32_t | Starting row index for this core |

### Compute Kernel
(No runtime arguments - all work parameters are compile-time)

### Writer Kernel
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer DRAM address |
| 1 | num_blocks | uint32_t | Number of tile blocks to write |
| 2 | start_stick_id | uint32_t | Starting output row index |

## Edge Cases
| Condition | Expected Behavior |
|-----------|-------------------|
| Single tile input (32×32) | Should work - minimum valid case |
| Large input (many tiles) | Should work - iterate through blocks |
| Height not multiple of 32 | Error: "Height must be multiple of TILE_HEIGHT (32)" |
| Width not multiple of 32 | Error: "Width must be multiple of TILE_WIDTH (32)" |
| Non-4D tensor | Error: "Input tensor must be 4D (NCHW)" |
| TILE_LAYOUT input | Error: "Input must be in ROW_MAJOR layout" |
| Sharded input | Error: "Input must be interleaved (not sharded)" |
| Host tensor | Error: "Input must be on device" |
| UINT8/INT8 dtype | Error: "Unsupported dtype" |

## Agent Handoff

This spec will be consumed by implementation agents. Each agent reads specific sections:

| Agent | Reads These Sections |
|-------|---------------------|
| **ttnn-operation-scaffolder** | API Specification, Input Tensor Requirements, Output Tensor Specification |
| **ttnn-factory-builder** | Circular Buffer Requirements, Work Distribution, Data Flow |
| **ttnn-kernel-dataflow** | Kernel Data Movement, Memory Access Patterns, Reader/Writer compile-time and runtime args |
| **ttnn-kernel-compute** | Compute Access Pattern, Mathematical Definition, Compute compile-time args |

The agents know HOW to implement; this spec defines WHAT to implement.

## Test Criteria

What behavior should be verified (agents decide how/when to test):

### Validation Behavior
- Non-4D tensor → error containing "must be 4D"
- Non-ROW_MAJOR layout → error containing "ROW_MAJOR"
- Sharded input → error containing "interleaved"
- Host tensor → error containing "device"
- Unsupported dtype → error containing "dtype"
- Height not multiple of 32 → error containing "TILE_HEIGHT"
- Width not multiple of 32 → error containing "TILE_WIDTH"

### Shape Behavior
- Output shape exactly matches input shape
- Output layout is ROW_MAJOR
- Output memory is INTERLEAVED

### Functional Behavior
- Single tile (32×32): output equals input exactly
- Multiple tiles (64×64): output equals input exactly
- Large tensor (128×256): output equals input exactly
- Numerical accuracy: bit-exact match with input (identity operation)

### Performance Behavior (informational)
- No hangs or deadlocks
- Completes within reasonable time bounds

## Extension Points

This template is designed for extension. To add compute operations:

### Adding a Unary Operation (e.g., ReLU)
```cpp
// In compute kernel, replace placeholder with:
cb_wait_front(CB_tiled, num_tiles_per_row);
for (uint32_t t = 0; t < num_tiles_per_row; t++) {
    acquire_dst();
    copy_tile(CB_tiled, t, 0);  // Load tile to DEST
    relu_tile(0);               // Apply ReLU
    pack_tile(0, CB_tiled, t);  // Pack back to CB
    release_dst();
}
// Then continue with untilize
```

### Adding a Reduction Operation
```cpp
// Modify CB_out sizing for reduced output shape
// In compute kernel:
cb_wait_front(CB_tiled, num_tiles_per_row);
reduce_init();
for (uint32_t t = 0; t < num_tiles_per_row; t++) {
    reduce_tile(CB_tiled, t, ...);  // Accumulate
}
// Pack reduced result to smaller output
```

### Adding Multi-Core Support
```cpp
// In program factory:
auto [ncores, all_cores, core_range, core_range_cliff,
      nblocks_per_core, nblocks_per_core_cliff] =
    split_blocks_for_tilize(grid_size, num_tile_rows);

// Set per-core runtime args for start_block, end_block
```

## Open Questions

1. **Double-buffering**: Should the template use double-buffered CBs for reader-compute and compute-writer overlap? Current design uses single-buffering for simplicity.

2. **Multi-core default**: Should multi-core be the default implementation rather than single-core? Trade-off between complexity and performance.

3. **Datatype support**: Should BFLOAT8_B and other compressed formats be supported? Would require additional CB configuration.

## References

### Reference Analyses
- Tilize: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_analysis.md`
- Untilize: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/untilize_analysis.md`

### Reference Implementations
- Pattern source: `ttnn/cpp/ttnn/operations/data_movement/permute/device/kernels/compute/transpose_xw_tiled.cpp`
- Factory reference: `ttnn/cpp/ttnn/operations/data_movement/permute/device/permute_tiled_program_factory.cpp`
- Reader reference: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/reader_unary_stick_layout_split_rows_interleaved.cpp`
- Writer reference: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/writer_unary_stick_layout_split_rows_multi_core.cpp`

### Helper Libraries
- Tilize: `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp`
- Untilize: `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp`

### DeepWiki Queries
| Query | Response Summary | How Used |
|-------|------------------|----------|
| "How to create intermediate CB for tilize-untilize" | Use dedicated CBIndex values (c_0, c_1, c_16), standard CB creation with CircularBufferConfig | CB architecture design |
| "CB best practices between tilize and untilize" | Dedicated intermediate CBs, buffering factor consideration, memory constraints | Sizing strategy |

### Documentation Consulted
- `ttnn/cpp/ttnn/operations/core/work_split/work_split_tilize.hpp` - Core distribution
- `ttnn/cpp/ttnn/operations/cb_utils.hpp` - CB creation helpers
- TensorAccessor API for memory addressing
