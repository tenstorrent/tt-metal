# reduce_avg_w_rm Functional Specification

## Overview
- **Operation Name**: reduce_avg_w_rm
- **Category**: reduction (fused format conversion + reduction)
- **Planning Mode**: Hybrid
- **Reference Operations**: tilize_multi_core_interleaved, untilize_multi_core
- **Reference Analyses**:
  - /localdev/dnijemcevic/tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_multi_core_interleaved_analysis.md (role: input_stage)
  - /localdev/dnijemcevic/tt-metal/ttnn/cpp/ttnn/operations/data_movement/untilize/device/factories/untilize_multi_core_analysis.md (role: output_stage)

## Mathematical Definition

### Formula
```
output[n, c, h, 0] = (1/W) * sum(w=0..W-1, input[n, c, h, w])
```

Where:
- `N, C, H, W` are the input tensor dimensions
- Output shape is `[N, C, H, 1]` (logical) with physical width of 32 (tile-aligned)
- The reduction computes the arithmetic mean along the width dimension

### Semantic Description
This operation takes a ROW_MAJOR tensor and computes the average of all elements along the width dimension for each (n, c, h) index. The pipeline is:
1. **Tilize**: Convert ROW_MAJOR input to TILE_LAYOUT (32x32 tiles)
2. **Reduce**: Compute SUM along width dimension with scaler = 1.0/W
3. **Untilize**: Convert TILE_LAYOUT output back to ROW_MAJOR

This is a fused operation that avoids intermediate DRAM writes between stages.

## API Specification

### Parameters
| Parameter | Type | Required | Valid Range | Default | Description |
|-----------|------|----------|-------------|---------|-------------|
| input_tensor | Tensor | Yes | - | - | Input tensor to reduce |
| output_mem_config | MemoryConfig | No | - | Default | Output memory configuration |
| compute_kernel_config | ComputeKernelConfig | No | - | Device default | Compute kernel configuration |

### Input Tensor Requirements

| Property | Requirement | Error Message Hint |
|----------|-------------|-------------------|
| Rank | 4 (N, C, H, W) | "Input tensor must have rank 4" |
| Layout | ROW_MAJOR | "Input tensor must be in ROW_MAJOR layout" |
| Memory Layout | INTERLEAVED | "Input tensor must have INTERLEAVED memory layout" |
| Data Type | BFLOAT16 | "Input tensor must be BFLOAT16" |
| Width | > 0 | "Input width must be positive" |
| Height | Multiple of 32 (tile-aligned) | "Input height must be multiple of TILE_HEIGHT (32)" |
| Width | Multiple of 32 (tile-aligned) | "Input width must be multiple of TILE_WIDTH (32)" |
| Buffer Type | DRAM or L1 | "Input must be in DRAM or L1" |

### Output Tensor Specification

| Property | Value |
|----------|-------|
| Shape | [N, C, H, 32] (physical), [N, C, H, 1] (logical) |
| Dtype | Same as input (BFLOAT16) |
| Layout | ROW_MAJOR |
| Memory Layout | INTERLEAVED |

**Shape Calculation**:
- Output N = Input N
- Output C = Input C
- Output H = Input H
- Output W = 32 (physical, tile-aligned) / 1 (logical)

**Note on Output Width**: The output width is physically 32 to maintain tile alignment for potential downstream operations. Only the first element (index 0) in each width row contains the valid average value. Elements 1-31 contain padding (zeros or garbage).

## Component Sources (Hybrid Mode)

This operation is composed from multiple references:

### Input Stage (from tilize_multi_core_interleaved)
| Component | Source | Modifications |
|-----------|--------|---------------|
| Reader kernel pattern | tilize.reader_unary_stick_layout_split_rows_interleaved | Read row-major sticks, need to also generate scaler tile |
| CB_in configuration | tilize.CB_0 (row-major sticks) | Size based on tiles_per_row (Wt) |
| Compute (tilize phase) | compute_kernel_lib::tilize() | Standard tilize from helper library |

### Compute Stage (new, with reduce_helpers)
| Component | Source | Modifications |
|-----------|--------|---------------|
| CB_tilized (intermediate) | New | Holds tilized tiles for reduce input |
| CB_scaler | From reduce_op patterns | Holds 1/W scaler tile for averaging |
| CB_reduced (intermediate) | New | Holds reduced tiles (1 tile per row) |
| Reduce operation | compute_kernel_lib::reduce<SUM, REDUCE_ROW>() | Standard reduce from helper library |

### Output Stage (from untilize_multi_core)
| Component | Source | Modifications |
|-----------|--------|---------------|
| Compute (untilize phase) | compute_kernel_lib::untilize() | Standard untilize from helper library |
| CB_out configuration | untilize.CB_16 (row-major output) | Size for 1 output tile per row |
| Writer kernel pattern | untilize.writer_unary_stick_layout_split_rows_multi_core | Write row-major sticks (width=32) |

### Interface Compatibility
| Interface | Component A | Component B | Format A | Format B | Compatible? |
|-----------|------------|-------------|----------|----------|-------------|
| Reader->Tilize | CB_in (sticks) | tilize() | Row-major sticks | Row-major sticks | Yes |
| Tilize->Reduce | CB_tilized | reduce() | TILE_LAYOUT | TILE_LAYOUT | Yes |
| Reduce->Untilize | CB_reduced | untilize() | TILE_LAYOUT | TILE_LAYOUT | Yes |
| Untilize->Writer | CB_out | Writer | Row-major (32-wide) | Row-major sticks | Yes |

### CB ID Resolution
| Logical CB | Source Ref | Original ID | Final ID | Notes |
|------------|-----------|-------------|----------|-------|
| CB_in (row-major input) | tilize | c_0 | c_0 | Input sticks from reader |
| CB_tilized | New | - | c_1 | Intermediate: tilized tiles |
| CB_scaler | reduce_op | c_2 | c_2 | Scaler tile (1/W) |
| CB_reduced | New | - | c_3 | Intermediate: reduced tiles |
| CB_out (row-major output) | untilize | c_16 | c_16 | Output sticks to writer |

## Design Decisions

### Decision 1: Fused Pipeline Architecture
- **Choice**: Fuse tilize-reduce-untilize in single kernel pipeline with intermediate CBs
- **Rationale**: Avoids DRAM round-trips between stages, better performance and lower memory bandwidth
- **Alternatives Considered**:
  - Three separate operations (tilize -> reduce -> untilize): Higher latency due to DRAM writes
  - Sharded intermediates: More complex, not needed for initial implementation
- **Tradeoffs**: More complex kernel logic, but significantly better performance

### Decision 2: Work Distribution Strategy
- **Choice**: Distribute work by tile rows (height dimension), similar to tilize/untilize patterns
- **Rationale**:
  - Each row of tiles is independent for W-reduction
  - Aligns with existing tilize/untilize core distribution
  - Good load balancing for typical tensor shapes
- **Alternatives Considered**:
  - Distribute by N*C batches: Coarser granularity, worse load balancing for small batches
  - 2D grid distribution: Unnecessary complexity for row-independent reduction
- **Tradeoffs**: Simple implementation, good parallelism for tall tensors

### Decision 3: Scaler Computation
- **Choice**: Compute scaler = 1.0/W on host, pass as compile-time argument to reader
- **Rationale**:
  - W is known at graph compile time
  - Matches existing reduce_op patterns
  - Reader generates scaler tile using generate_reduce_scaler()
- **Alternatives Considered**:
  - Compute scaler on device: Unnecessary overhead
  - Use reduce SUM then divide: Extra operation, less efficient
- **Tradeoffs**: Requires recompile if W changes, but W is typically static

### Decision 4: Output Width Handling
- **Choice**: Physical output width = 32 (one tile), logical width = 1
- **Rationale**:
  - Tile-aligned output enables downstream tile-based operations
  - Only first column has valid data after W-reduction
  - Matches reduce_op output format
- **Alternatives Considered**:
  - Output width = 1 (non-tile-aligned): Would require special handling for downstream ops
- **Tradeoffs**: Some wasted memory for padding, but maintains system compatibility

### Decision 5: Single Compute Kernel with Three Phases
- **Choice**: One compute kernel that does tilize->reduce->untilize in sequence per block
- **Rationale**:
  - Simplifies CB synchronization
  - Each phase uses helper library functions
  - Data stays in L1 between phases
- **Alternatives Considered**:
  - Separate compute kernels per phase: Would need inter-kernel sync, more complex
- **Tradeoffs**: Slightly more complex kernel, but cleaner data flow

### Decision 6: Streaming Input Mode for Reduce
- **Choice**: Use ReduceInputMode::STREAMING for the reduce helper
- **Rationale**:
  - Tiles arrive one-at-a-time from tilize phase
  - Most memory-efficient pattern
  - Compatible with single-buffered CBs
- **Alternatives Considered**:
  - STREAMING_BATCHED: Would require larger CB to hold entire row
- **Tradeoffs**: Slightly lower throughput, but safer and simpler

## Work Distribution

### Work Unit Definition
- **Work unit**: One row of tiles (all Wt tiles at same H index)
- **Processing**: Each work unit produces one output tile (after W-reduction)
- **Total work units**: NC * Ht (batch * channel * height_in_tiles)

### Parallelization Strategy
- **Grid**: 1D grid (linearized core assignment)
- **Work per core**: `ceil(total_rows / num_cores)` rows per core
- **Load balancing**: Near-equal distribution with cliff core handling (last core may have fewer rows)

### Core Assignment Algorithm
```
nrows = NC * Ht
num_cores = min(nrows, grid_size.x * grid_size.y)
rows_per_core = ceil(nrows / num_cores)
rows_per_cliff_core = nrows % rows_per_core (if non-zero)
```

## Data Flow

### High-Level Flow
```
DRAM (Row-Major) --> Reader --> CB_in (sticks)
                                    |
                                    v
                            Compute: tilize()
                                    |
                                    v
                             CB_tilized (tiles)
                                    |
                    CB_scaler ----> v
                        |    Compute: reduce()
                        |           |
                        v           v
                    (scaler)   CB_reduced (1 tile/row)
                                    |
                                    v
                            Compute: untilize()
                                    |
                                    v
                              CB_out (sticks)
                                    |
                                    v
                  Writer --> DRAM (Row-Major)
```

### Kernel Data Movement

| Kernel | Core | NOC | Actual Function |
|--------|------|-----|-----------------|
| Reader | RISCV_0 | NOC0 | Read row-major input sticks from DRAM, generate scaler tile |
| Compute | RISCV_2 | N/A | tilize -> reduce -> untilize pipeline |
| Writer | RISCV_1 | NOC1 | Write row-major output sticks to DRAM |

### Detailed Per-Block Flow (one row of tiles)

**Reader**:
1. Generate scaler tile (1/W) into CB_scaler (once per kernel launch)
2. For each block (tile row):
   - Reserve CB_in for Wt tiles worth of sticks (32 * Wt rows)
   - Read 32 row-major sticks from DRAM (TILE_HEIGHT sticks)
   - Push to CB_in

**Compute**:
1. Initialize all phases: tilize_init, reduce_init, untilize_init
2. Wait for scaler tile in CB_scaler
3. For each block (tile row):
   - **Tilize phase**:
     - Wait for sticks in CB_in
     - Reserve CB_tilized for Wt tiles
     - tilize_block(CB_in, Wt, CB_tilized)
     - Push CB_tilized, pop CB_in
   - **Reduce phase**:
     - For each of Wt tiles:
       - Wait for 1 tile in CB_tilized
       - reduce_tile(CB_tilized, CB_scaler, dst=0)
       - Pop CB_tilized
     - Reserve CB_reduced for 1 tile
     - pack_tile(0, CB_reduced)
     - Push CB_reduced
   - **Untilize phase**:
     - Wait for 1 tile in CB_reduced
     - Reserve CB_out for 1 tile worth of sticks
     - untilize_block(CB_reduced, 1, CB_out)
     - Push CB_out, pop CB_reduced
4. Uninitialize all phases

**Writer**:
1. For each block (tile row):
   - Wait for sticks in CB_out
   - Write 32 row-major sticks to DRAM (output width = 32)
   - Pop CB_out

### Circular Buffer Requirements

| CB ID | Name | Purpose | Producer | Consumer | Sizing Strategy | Lifetime |
|-------|------|---------|----------|----------|-----------------|----------|
| c_0 | cb_in | Input row-major sticks | Reader | Compute (tilize) | Wt tiles * tile_size (single-buffered per block) | Block |
| c_1 | cb_tilized | Tilized tiles (intermediate) | Compute (tilize) | Compute (reduce) | Wt tiles * tile_size (single-buffered) | Block |
| c_2 | cb_scaler | Scaler tile (1/W) | Reader | Compute (reduce) | 1 tile | Kernel |
| c_3 | cb_reduced | Reduced tiles (intermediate) | Compute (reduce) | Compute (untilize) | 1 tile (single-buffered) | Block |
| c_16 | cb_out | Output row-major sticks | Compute (untilize) | Writer | 1 tile * tile_size (single-buffered) | Block |

**CB Sizing Formulas**:
- `cb_in_size = Wt * tile_size` (holds one row of input tiles worth of sticks)
- `cb_tilized_size = Wt * tile_size` (holds one row of tilized tiles)
- `cb_scaler_size = 1 * tile_size` (single scaler tile)
- `cb_reduced_size = 1 * tile_size` (one reduced tile per row)
- `cb_out_size = 1 * tile_size` (one output tile worth of sticks)

## Memory Access Patterns

### RISCV_0 ("reader" / BRISC) Access
1. **Scaler generation** (once):
   - Read zeros from MEM_ZEROS_BASE to fill tile
   - Write scaler value pattern into tile
   - Push to CB_scaler

2. **Input reading** (per block):
   - For each of TILE_HEIGHT (32) rows in block:
     - Compute NoC address for stick using TensorAccessor
     - Issue noc_async_read for full stick width (W * element_size bytes)
   - Barrier after all 32 reads
   - Push block to CB_in

### RISCV_1 ("writer" / NCRISC) Access
1. **Output writing** (per block):
   - Wait for CB_out to have data
   - For each of TILE_HEIGHT (32) rows:
     - Get L1 read pointer
     - Compute NoC address for output stick using TensorAccessor
     - Issue noc_async_write for output stick (32 * element_size bytes)
   - Barrier after all 32 writes
   - Pop CB_out

### Compute Access
1. **Tilize** (per block):
   - cb_wait_front(cb_in, Wt) - wait for input sticks
   - cb_reserve_back(cb_tilized, Wt) - reserve output space
   - tilize_block(cb_in, Wt, cb_tilized) - transform
   - cb_push_back(cb_tilized, Wt)
   - cb_pop_front(cb_in, Wt)

2. **Reduce** (per block, processes Wt tiles):
   - tile_regs_acquire()
   - For wt in 0..Wt-1:
     - cb_wait_front(cb_tilized, 1)
     - reduce_tile(cb_tilized, cb_scaler, 0, 0, 0)
     - cb_pop_front(cb_tilized, 1)
   - cb_reserve_back(cb_reduced, 1)
   - tile_regs_commit(), tile_regs_wait()
   - pack_tile(0, cb_reduced)
   - tile_regs_release()
   - cb_push_back(cb_reduced, 1)

3. **Untilize** (per block):
   - cb_wait_front(cb_reduced, 1)
   - cb_reserve_back(cb_out, 1)
   - untilize_block(cb_reduced, 1, cb_out) or pack_untilize (auto-selected)
   - cb_push_back(cb_out, 1)
   - cb_pop_front(cb_reduced, 1)

## Compile-Time Arguments

### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | packed_scaler_value | uint32_t | Scaler (1/W) packed as two bfloat16 |
| 1 | stick_size | uint32_t | Input stick size in bytes (W * sizeof(bfloat16)) |
| 2+ | TensorAccessorArgs | varies | Input tensor accessor configuration |

### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_blocks_per_core | uint32_t | Number of tile rows for this core |
| 1 | Wt | uint32_t | Width in tiles (tiles per row) |

### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_out | uint32_t | Output CB ID (c_16) |
| 1 | output_stick_size | uint32_t | Output stick size in bytes (32 * sizeof(bfloat16)) |
| 2+ | TensorAccessorArgs | varies | Output tensor accessor configuration |

## Runtime Arguments

### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Input buffer DRAM address |
| 1 | num_sticks | uint32_t | Total input sticks to read (num_blocks * TILE_HEIGHT) |
| 2 | start_stick_id | uint32_t | Starting stick index for this core |

### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_blocks | uint32_t | Number of blocks (tile rows) to process |

### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Output buffer DRAM address |
| 1 | num_sticks | uint32_t | Total output sticks to write (num_blocks * TILE_HEIGHT) |
| 2 | start_stick_id | uint32_t | Starting output stick index for this core |

## Edge Cases

| Condition | Expected Behavior |
|-----------|-------------------|
| Single tile width (W=32) | Reduce produces average of 32 elements per row, valid operation |
| Large width (W=1024+) | Works correctly, Wt tiles processed sequentially |
| Single row (H=32) | Single output row, one tile, valid operation |
| Large batch (N*C large) | Distributed across many cores, good parallelism |
| Minimum tensor (1,1,32,32) | Single tile in, single tile out, reduces to average |
| Non-tile-aligned input | Error: "Input dimensions must be multiples of 32" |
| Non-BFLOAT16 input | Error: "Input tensor must be BFLOAT16" |
| Non-ROW_MAJOR input | Error: "Input tensor must be in ROW_MAJOR layout" |
| Sharded input | Error: "Input tensor must have INTERLEAVED memory layout" |

## Agent Handoff

This spec will be consumed by implementation agents. Each agent reads specific sections:

| Agent | Reads These Sections |
|-------|---------------------|
| **ttnn-operation-scaffolder** | API Specification, Input Tensor Requirements, Output Tensor Specification |
| **ttnn-factory-builder** | Circular Buffer Requirements, Work Distribution, Data Flow, Component Sources |
| **ttnn-kernel-designer** | Mathematical Definition, Data Flow, Circular Buffer Requirements, Memory Access Patterns |
| **ttnn-kernel-writer** | Kernel Design Document (from kernel-designer), Compile-Time/Runtime Arguments |

The agents know HOW to implement; this spec defines WHAT to implement.

## Test Criteria

What behavior should be verified (agents decide how/when to test):

### Validation Behavior
- Non-rank-4 tensor -> error containing "Input tensor must have rank 4"
- Non-ROW_MAJOR layout -> error containing "Input tensor must be in ROW_MAJOR layout"
- Non-INTERLEAVED memory -> error containing "Input tensor must have INTERLEAVED memory layout"
- Non-BFLOAT16 dtype -> error containing "Input tensor must be BFLOAT16"
- Non-tile-aligned dimensions -> error containing "must be multiple of"

### Shape Behavior
- Output shape [N, C, H, 32] matches formula for input [N, C, H, W]
- Output has ROW_MAJOR layout
- Output has INTERLEAVED memory layout

### Functional Behavior
- Single tile (32x32): output[0,0,h,0] = mean(input[0,0,h,:]) for all h
- Multi-tile width: correct average across all Wt tiles
- Multi-batch: each batch processed independently and correctly
- Numerical accuracy: |output - torch.mean(input, dim=-1)| < tolerance (1e-2 for bfloat16)

### Performance Criteria (Optional)
- Multi-core utilization: operations scale with available cores
- No DRAM writes for intermediate data (tilized/reduced tensors stay in L1)

## Open Questions

1. **Double buffering**: Should we implement double-buffered CBs for better throughput? Current design uses single-buffered for simplicity. Could add as optimization later.

2. **Output width padding**: Should the output be logically [N,C,H,1] with physical storage handling done internally, or should we expose [N,C,H,32] to users? Current design exposes [N,C,H,32] for tile alignment.

3. **FP32 accumulation**: Should we enable FP32 destination accumulation for better precision during the reduction? Current design follows input dtype. Could add compute_kernel_config option.

4. **Sharded input support**: Should we add support for sharded input tensors? Current design requires INTERLEAVED. Could be added as separate program factory variant.

## References
- Reference analyses:
  - /localdev/dnijemcevic/tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_multi_core_interleaved_analysis.md
  - /localdev/dnijemcevic/tt-metal/ttnn/cpp/ttnn/operations/data_movement/untilize/device/factories/untilize_multi_core_analysis.md
- Helper libraries:
  - /localdev/dnijemcevic/tt-metal/ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp
  - /localdev/dnijemcevic/tt-metal/ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp
  - /localdev/dnijemcevic/tt-metal/ttnn/cpp/ttnn/kernel_lib/reduce_helpers.hpp
- Existing reduce patterns:
  - /localdev/dnijemcevic/tt-metal/ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_op_multi_core_w_program_factory.cpp
  - /localdev/dnijemcevic/tt-metal/ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce_w.cpp
- Scaler generation:
  - /localdev/dnijemcevic/tt-metal/ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp
- DeepWiki queries:
  - "How is the scaler value (e.g., 1/W for average) computed and passed to reduce operations?"
- Documentation consulted:
  - METALIUM_GUIDE.md (architecture, CB semantics)
  - tech_reports/tensor_layouts/tensor_layouts.md (row-major vs tile layout)
