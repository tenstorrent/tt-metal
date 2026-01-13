# Row Mean Sub Square Reduce Functional Specification

## Overview
- **Operation Name**: `row_mean_sub_square_reduce` (alias: `variance_w`)
- **Category**: reduction
- **Planning Mode**: Hybrid
- **Reference Operation(s)**: tilize_multi_core_interleaved, reduce_multi_core_w, untilize_multi_core
- **Reference Analyses**:
  - `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_multi_core_interleaved_analysis.md` (role: input_stage)
  - `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_multi_core_w_analysis.md` (role: compute_core)
  - `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/data_movement/untilize/device/factories/untilize_multi_core_analysis.md` (role: output_stage)

## Mathematical Definition

### Formula
```
mean[n,c,h] = (1/W) * sum(input[n,c,h,w] for w in 0..W-1)
diff[n,c,h,w] = input[n,c,h,w] - mean[n,c,h]
variance[n,c,h] = (1/W) * sum(diff[n,c,h,w]^2 for w in 0..W-1)

output[n,c,h,0:TILE_WIDTH] = variance[n,c,h] (padded to TILE_WIDTH=32)
```

### Semantic Description
This operation computes the variance along the width (W) dimension of an input tensor. For each position (n, c, h):
1. Compute the mean of all W elements: E[x]
2. Subtract the mean from each element: (x - E[x])
3. Square each difference: (x - E[x])^2
4. Compute the mean of the squared differences: E[(x - E[x])^2]

The result is the variance at each (n, c, h) position, representing the spread of values along W.

## API Specification

### Parameters
| Parameter | Type | Required | Valid Range | Default | Description |
|-----------|------|----------|-------------|---------|-------------|
| input_tensor | Tensor | Yes | - | - | Input tensor in ROW_MAJOR layout |
| memory_config | MemoryConfig | No | DRAM/L1 | Input memory config | Output memory configuration |
| output_dtype | DataType | No | BFLOAT16 | Input dtype | Output data type |

### Input Tensor Requirements

| Property | Requirement | Error Message Hint |
|----------|-------------|-------------------|
| Rank | Must be 4D [N, C, H, W] | "Input tensor must be 4D" |
| Layout | ROW_MAJOR | "Input must be in ROW_MAJOR layout" |
| Dtype | BFLOAT16 | "Input dtype must be BFLOAT16" |
| Memory | INTERLEAVED on DRAM | "Input must be DRAM interleaved" |
| Device | Must be on device | "Input tensor must be on device" |
| Width | W >= 1 | "Width must be at least 1" |
| Alignment | W must be divisible by TILE_WIDTH (32) or will be padded | "Width padded to TILE_WIDTH" |

### Output Tensor Specification
- **Shape**: [N, C, H, TILE_WIDTH] where TILE_WIDTH = 32
  - Logical width is 1 (the variance value), but padded to tile width
- **Dtype**: BFLOAT16 (or user-specified output_dtype)
- **Layout**: ROW_MAJOR
- **Memory layout**: INTERLEAVED (same as input or user-specified)

## Component Sources (Hybrid Mode)

This operation is composed from multiple references:

### Input Stage (from tilize_multi_core_interleaved)
| Component | Source | Modifications |
|-----------|--------|---------------|
| Reader kernel pattern | tilize.reader_unary_stick_layout | Reads row-major sticks using TensorAccessor |
| CB_rm_in configuration | tilize.CB_c_0 | For row-major input staging, sized for ntiles_per_row |
| Compute (tilize phase) | tilize.compute | Uses tilize_block helper to convert stick to tile format |

### Compute Stage (from reduce_multi_core_w + new compute)
| Component | Source | Modifications |
|-----------|--------|---------------|
| Scaler generation | reduce_w.generate_reduce_scaler | Creates 1/W scaler tile for mean computation |
| CB_scaler | reduce_w.CB_c_2 | Holds scaler tile, persists for entire program |
| Reduce (REDUCE_ROW) | reduce_w.compute | Modified to compute mean first, then variance |
| CB_mean | New | Stores intermediate mean tiles for subtraction |
| Sub/Square/Reduce | New | Custom compute: subtract mean, square, reduce again |

### Output Stage (from untilize_multi_core)
| Component | Source | Modifications |
|-----------|--------|---------------|
| Compute (untilize phase) | untilize.compute | Uses untilize_block/pack_untilize helper |
| CB_out configuration | untilize.CB_c_16 | For row-major output staging |
| Writer kernel pattern | untilize.writer_stick_layout | Writes row-major sticks using TensorAccessor |

### Interface Compatibility

| Interface | From Component | To Component | Format A | Format B | Compatible? |
|-----------|---------------|--------------|----------|----------|-------------|
| Reader->Compute(tilize) | tilize.reader | tilize.compute | Row-major sticks | Row-major sticks in CB | Yes |
| Tilize->Reduce | tilize.compute | reduce.compute | Tiled | Tiled | Yes |
| Reduce->Untilize | reduce.compute | untilize.compute | Tiled (1 tile wide) | Tiled | Yes |
| Untilize->Writer | untilize.compute | untilize.writer | Row-major in CB | Row-major sticks | Yes |

### CB ID Resolution

| Logical CB | Source Ref | Original ID | Final ID | Notes |
|------------|-----------|-------------|----------|-------|
| CB_rm_in | tilize | c_0 | c_0 | Row-major input staging |
| CB_tilized | tilize | c_16 | c_1 | Tilized input, feed to compute |
| CB_scaler | reduce_w | c_2 | c_2 | Scaler tile (1/W), persists |
| CB_mean | New | - | c_3 | Mean tiles for broadcast subtract |
| CB_intermediate | New | - | c_4 | Squared differences |
| CB_out_tiled | untilize | c_0 | c_5 | Variance tiles before untilize |
| CB_rm_out | untilize | c_16 | c_16 | Row-major output staging |

## Design Decisions

### Decision 1: Single-Pass vs Multi-Pass Computation
- **Choice**: Single-pass through data with intermediate CB storage for mean
- **Rationale**: Avoids reading input twice from DRAM, which would double memory bandwidth. Mean can be stored temporarily in L1 CB while processing the tile row.
- **Alternatives Considered**:
  - Two-pass (read once for mean, again for variance): Doubles DRAM reads
  - Online/Welford algorithm: More complex, similar memory requirements
- **Tradeoffs**: Requires additional CB for mean storage, but saves DRAM bandwidth

### Decision 2: Work Distribution Granularity
- **Choice**: Parallelize over Ht (tile rows), each core processes complete tile-rows
- **Rationale**: Each row of tiles is independent for variance computation. Within a row, all Wt tiles contribute to one output tile, making the row the natural work unit.
- **Alternatives Considered**:
  - Parallelize over individual tiles: Would require inter-core communication for reduction
  - Parallelize over NC batches: Less parallelism for small batch sizes
- **Tradeoffs**: Good load balancing when Ht is large; may underutilize cores for small Ht

### Decision 3: CB Sizing Strategy
- **Choice**: CB_tilized and CB_intermediate sized for Wt tiles (full row), others double-buffered for overlap
- **Rationale**: Need full row of input tiles to compute mean before subtraction. Double buffering on output path allows compute/write overlap.
- **Alternatives Considered**:
  - Streaming single tile at a time: Would require multiple passes over data
  - Full tensor in L1: Not feasible for large tensors
- **Tradeoffs**: L1 memory usage scales with W; may limit max tensor width

### Decision 4: Compute Kernel Architecture
- **Choice**: Fused compute kernel performing tilize, reduce(mean), sub, square, reduce(variance), untilize
- **Rationale**: Reduces data movement between kernels, all operations on same data before writing to output CB
- **Alternatives Considered**:
  - Separate kernels for each phase: More modular but higher CB pressure
  - Multiple programs chained: Would require intermediate DRAM writes
- **Tradeoffs**: Complex compute kernel, but maximizes data locality

### Decision 5: Scaler Format
- **Choice**: Single scaler value = 1/W packed as bfloat16, used for both mean computations
- **Rationale**: Same scaler applies to both E[x] and E[(x-E[x])^2] since both divide by W
- **Alternatives Considered**:
  - Separate scalers for each reduction: Unnecessary duplication
  - No scaler (divide after): Hardware reduction with scaler is more efficient
- **Tradeoffs**: None significant; reusing scaler is strictly better

### Decision 6: Mean Broadcast Strategy
- **Choice**: Use sub_tiles_bcast_scalar with mean tile for subtraction
- **Rationale**: Mean result is a single value per row (stored at tile[0,0]). Broadcast scalar subtraction efficiently applies this to all Wt input tiles.
- **Alternatives Considered**:
  - Copy mean to all positions manually: Unnecessary work
  - Column broadcast: Wrong dimension
- **Tradeoffs**: Requires mean to be in scalar position in tile

## Work Distribution

### Work Unit Definition
One **work unit** = one tile-row:
- Read Wt input tiles (one horizontal strip)
- Tilize all Wt tiles
- Reduce to mean (1 tile)
- Subtract mean from all Wt tiles, square each
- Reduce squared differences to variance (1 tile)
- Untilize variance tile
- Write 1 output tile (TILE_HEIGHT sticks)

### Parallelization Strategy
- **Grid**: Dynamic based on total tile-rows (NC * Ht)
- **Work per core**: `ceil(NC * Ht / num_cores)` tile-rows
- **Load balancing**: Two-group split (group_1 gets +1 row if uneven, same as reduce_w)

### Core Distribution
| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (linear enumeration) |
| **Grid dimensions** | Up to compute_with_storage_grid_size |
| **Total cores** | min(NC * Ht, max_cores) |
| **Work per core** | num_rows_per_core tile-rows |
| **Load balancing** | Two-group split for remainder |

## Data Flow

### High-Level Flow
```
DRAM (Row-Major)          L1 Circular Buffers           DRAM (Row-Major)
     |                           |                           |
     | [Wt tiles as sticks]      |                           |
     |-------------------------->| CB_rm_in (c_0)            |
     |                           |                           |
     |                           | tilize                    |
     |                           |-------------------------->| CB_tilized (c_1)
     |                           |                           |
     |                           | reduce_row (mean)         |
     |                           |-------------------------->| CB_mean (c_3)
     |                           |                           |
     |                           | for each tile in row:     |
     |                           |   sub_bcast_scalar(tile, mean)
     |                           |   square_tile             |
     |                           |-------------------------->| CB_intermediate (c_4)
     |                           |                           |
     |                           | reduce_row (variance)     |
     |                           |-------------------------->| CB_out_tiled (c_5)
     |                           |                           |
     |                           | untilize                  |
     |                           |-------------------------->| CB_rm_out (c_16)
     |                           |                           |
     |                           | [1 tile as sticks]        |
     |<--------------------------|---------------------------|
```

### Kernel Data Movement

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader | RISCV_0 (BRISC) | NOC0 | DRAM (row-major sticks) | CB_rm_in (c_0) | Read Wt*32 sticks per tile-row, generate_reduce_scaler once |
| compute | TRISC (Unpack/Math/Pack) | N/A | CB_rm_in, CB_scaler | CB_rm_out | tilize, reduce(mean), sub_bcast, square, reduce(var), untilize |
| writer | RISCV_1 (NCRISC) | NOC1 | CB_rm_out (c_16) | DRAM (row-major sticks) | Write 32 sticks per output tile |

### Circular Buffer Requirements

| CB ID | Name | Purpose | Producer | Consumer | Sizing Strategy | Lifetime |
|-------|------|---------|----------|----------|-----------------|----------|
| c_0 | CB_rm_in | Row-major input staging | Reader | Compute | Wt tiles (full row width) | Block |
| c_1 | CB_tilized | Tilized input tiles | Compute (tilize) | Compute (reduce) | Wt tiles | Block |
| c_2 | CB_scaler | Scaler tile (1/W) | Reader (once) | Compute | 1 tile | Program |
| c_3 | CB_mean | Mean tile for broadcast | Compute (reduce) | Compute (sub) | 1 tile | Block |
| c_4 | CB_intermediate | Squared differences | Compute (square) | Compute (reduce) | Wt tiles | Block |
| c_5 | CB_out_tiled | Variance tile (tiled) | Compute (reduce) | Compute (untilize) | 1 tile | Block |
| c_16 | CB_rm_out | Row-major output staging | Compute (untilize) | Writer | 2 tiles (double buffer) | Block |

**Sizing Formulas**:
- `Wt = ceil(W / TILE_WIDTH)` = tiles along width
- `tile_size = tt::tile_size(DataFormat::Float16_b)` = 2048 bytes for bfloat16
- `CB_rm_in pages = Wt`, `page_size = tile_size`
- `CB_tilized pages = Wt`, `page_size = tile_size`
- `CB_scaler pages = 1`, `page_size = tile_size`
- `CB_mean pages = 1`, `page_size = tile_size`
- `CB_intermediate pages = Wt`, `page_size = tile_size`
- `CB_out_tiled pages = 1`, `page_size = tile_size`
- `CB_rm_out pages = 2`, `page_size = tile_size` (double-buffered)

## Memory Access Patterns

### RISCV_0 ("reader" / BRISC) Access

**Scaler Generation (once at start)**:
1. Call `generate_reduce_scaler(CB_scaler, packed_scaler_value)` to fill scaler tile
2. Scaler = 1/W packed as two bfloat16 in uint32

**Row-Major Stick Reading (per tile-row)**:
1. Pre-compute NOC addresses for TILE_HEIGHT (32) consecutive sticks
2. For each of Wt tiles in the row:
   - Read 32 sticks (each stick = W * element_size bytes, but we read full row)
   - Actually: read stick_size = padded_W * element_size per stick
3. Data layout in CB: 32 rows x (Wt * TILE_WIDTH) elements

**Access Pattern**:
```
for each tile_row (0 to Ht-1):
    base_stick_id = tile_row * TILE_HEIGHT
    for j in 0..TILE_HEIGHT-1:
        noc_addr = get_noc_addr(base_stick_id + j, tensor_accessor)
        noc_async_read(noc_addr, l1_addr, row_width_bytes)
    noc_async_read_barrier()
    cb_push_back(CB_rm_in, Wt)
```

### RISCV_1 ("writer" / NCRISC) Access

**Row-Major Stick Writing (per output tile)**:
1. Wait for untilized data in CB_rm_out
2. For each row (0 to TILE_HEIGHT-1):
   - Calculate output page ID = (tile_row * TILE_HEIGHT + row)
   - Write stick_size = TILE_WIDTH * element_size bytes
3. Barrier after all rows written

**Access Pattern**:
```
for each output_tile:
    cb_wait_front(CB_rm_out, 1)
    base_l1_addr = get_read_ptr(CB_rm_out)

    for row in 0..TILE_HEIGHT-1:
        output_page_id = tile_row_index * TILE_HEIGHT + row
        l1_addr = base_l1_addr + row * TILE_WIDTH * element_size
        noc_addr = get_noc_addr(output_page_id, tensor_accessor)
        noc_async_write(l1_addr, noc_addr, TILE_WIDTH * element_size)

    noc_async_write_barrier()
    cb_pop_front(CB_rm_out, 1)
```

### Compute Access

**Phase 1: Tilize**
```
cb_wait_front(CB_rm_in, Wt)
cb_reserve_back(CB_tilized, Wt)
tilize_block(CB_rm_in, Wt, CB_tilized)
cb_push_back(CB_tilized, Wt)
cb_pop_front(CB_rm_in, Wt)
```

**Phase 2: Reduce to Mean**
```
cb_wait_front(CB_scaler, 1)  // Scaler ready (once)
reduce_init<REDUCE_OP::SUM, REDUCE_DIM::REDUCE_ROW>(CB_tilized, CB_scaler, CB_mean)
acquire_dst()
for wt in 0..Wt-1:
    cb_wait_front(CB_tilized, 1)
    reduce_tile<REDUCE_OP::SUM, REDUCE_DIM::REDUCE_ROW>(CB_tilized, CB_scaler, 0, 0, 0)
    // OR: matmul_tiles(CB_tilized, CB_scaler, 0, 0, 0)  // REDUCE_ROW_SUM_VIA_MM
    cb_pop_front(CB_tilized, 1)
cb_reserve_back(CB_mean, 1)
pack_tile(0, CB_mean)
cb_push_back(CB_mean, 1)
release_dst()
```

**Phase 3: Subtract Mean, Square (per tile)**
```
// Re-tilize input (or double-buffer to keep tilized)
// For each tile:
cb_wait_front(CB_mean, 1)  // Mean ready for all tiles in row
sub_tiles_bcast_scalar_init_short()
for wt in 0..Wt-1:
    cb_wait_front(CB_tilized, 1)  // Need input again
    acquire_dst()
    sub_tiles_bcast_scalar(CB_tilized, CB_mean, 0, 0, 0)  // diff = x - mean
    square_tile(0)  // sq = diff^2
    cb_reserve_back(CB_intermediate, 1)
    pack_tile(0, CB_intermediate)
    cb_push_back(CB_intermediate, 1)
    release_dst()
    cb_pop_front(CB_tilized, 1)
cb_pop_front(CB_mean, 1)
```

**Phase 4: Reduce Squared Differences to Variance**
```
reduce_init<REDUCE_OP::SUM, REDUCE_DIM::REDUCE_ROW>(CB_intermediate, CB_scaler, CB_out_tiled)
acquire_dst()
for wt in 0..Wt-1:
    cb_wait_front(CB_intermediate, 1)
    reduce_tile<REDUCE_OP::SUM, REDUCE_DIM::REDUCE_ROW>(CB_intermediate, CB_scaler, 0, 0, 0)
    cb_pop_front(CB_intermediate, 1)
cb_reserve_back(CB_out_tiled, 1)
pack_tile(0, CB_out_tiled)
cb_push_back(CB_out_tiled, 1)
release_dst()
```

**Phase 5: Untilize**
```
cb_wait_front(CB_out_tiled, 1)
cb_reserve_back(CB_rm_out, 1)
untilize_block(CB_out_tiled, 1, CB_rm_out)
cb_push_back(CB_rm_out, 1)
cb_pop_front(CB_out_tiled, 1)
```

## Compile-Time Arguments

### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | stick_size | uint32_t | Size of one input row in bytes (padded_W * element_size) |
| 1 | packed_scaler_value | uint32_t | 1/W packed as two bfloat16 |
| 2+ | TensorAccessorArgs | varies | Input buffer addressing (bank info, shapes) |

### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_cb_index | uint32_t | Output CB ID (c_16) |
| 1 | output_stick_size | uint32_t | Size of one output row in bytes (TILE_WIDTH * element_size) |
| 2 | tile_height | uint32_t | TILE_HEIGHT (32) |
| 3+ | TensorAccessorArgs | varies | Output buffer addressing |

### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | Wt | uint32_t | Number of tiles along W dimension |
| 1 | num_rows_per_core | uint32_t | Tile-rows to process on this core |

## Runtime Arguments

### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer DRAM address |
| 1 | num_sticks | uint32_t | Total sticks to read (num_rows_per_core * TILE_HEIGHT) |
| 2 | num_tiles_per_row | uint32_t | Wt (tiles per row) |
| 3 | row_width_bytes | uint32_t | padded_W * element_size |
| 4 | start_stick_id | uint32_t | First stick ID for this core |

### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer DRAM address |
| 1 | num_output_tiles | uint32_t | Output tiles to write (num_rows_per_core) |
| 2 | start_output_stick_id | uint32_t | First output stick ID for this core |

### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_rows_this_core | uint32_t | Tile-rows assigned to this core |

## Edge Cases

| Condition | Expected Behavior |
|-----------|-------------------|
| W = TILE_WIDTH (32) | Single tile per row, Wt=1, simplest case |
| W = 1 | Variance is 0 (no spread), scaler = 1/1 = 1 |
| W not divisible by 32 | Pad to next multiple of 32, compute on padded |
| Single tile row (Ht=1) | Only one core needed, no distribution |
| Large H (many rows) | Good parallelism, distribute across cores |
| H * NC > num_cores | Multiple rows per core, cliff core handling |
| All elements equal | Mean = element, variance = 0 |
| Empty tensor (W=0) | Error: "Width must be at least 1" |
| Non-4D input | Error: "Input tensor must be 4D" |
| Wrong layout | Error: "Input must be in ROW_MAJOR layout" |

## Agent Handoff

This spec will be consumed by implementation agents. Each agent reads specific sections:

| Agent | Reads These Sections |
|-------|---------------------|
| **ttnn-operation-scaffolder** | API Specification, Input Tensor Requirements, Output Tensor Specification |
| **ttnn-factory-builder** | Circular Buffer Requirements, Work Distribution, Data Flow, Component Sources |
| **ttnn-kernel-dataflow** | Kernel Data Movement, Memory Access Patterns, Component Sources |
| **ttnn-kernel-compute** | Compute Access, Mathematical Definition, Component Sources |

The agents know HOW to implement; this spec defines WHAT to implement.

## Test Criteria

What behavior should be verified (agents decide how/when to test):

### Validation Behavior
- Wrong tensor rank -> error containing "must be 4D"
- Wrong layout -> error containing "ROW_MAJOR"
- Unsupported dtype -> error containing "BFLOAT16"
- Non-device tensor -> error containing "must be on device"
- Empty width (W=0) -> error containing "Width must be"

### Shape Behavior
- Output shape = [N, C, H, TILE_WIDTH] where TILE_WIDTH=32
- Output is ROW_MAJOR layout
- Output is INTERLEAVED memory

### Functional Behavior
- Single tile width (W=32): variance computed correctly
- Multi-tile width (W=64, 128, ...): variance matches PyTorch
- Uniform values: variance = 0
- Known distribution: variance matches expected
- Numerical accuracy: rtol=1e-2, atol=1e-3 vs torch.var(dim=-1, keepdim=True)

### PyTorch Reference
```python
def reference_variance_w(input_tensor):
    # input: [N, C, H, W]
    variance = torch.var(input_tensor, dim=-1, keepdim=True, unbiased=False)
    # variance: [N, C, H, 1]
    # Pad to TILE_WIDTH=32
    output = torch.nn.functional.pad(variance, (0, 31))  # [N, C, H, 32]
    return output
```

## Open Questions

1. **Double-buffer tilized input?**: Should we keep the tilized input in a separate double-buffered CB to avoid re-tilizing for the subtract phase? This would increase L1 usage but avoid recomputation.
   - **Current decision**: Re-read and re-tilize to save L1, but may reconsider if performance is critical.

2. **FP32 accumulation for reduce?**: Should we enable FP32 destination accumulation for the reduction operations to improve numerical accuracy?
   - **Recommendation**: Enable `fp32_dest_acc_en = true` for better precision, especially for large W.

3. **Output padding handling**: The output has logical width 1 but physical width 32. Should we zero-pad the unused 31 elements, or leave them undefined?
   - **Recommendation**: Leave undefined (default behavior of untilize), document that only element 0 is valid.

## References
- Reference analyses:
  - `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_multi_core_interleaved_analysis.md`
  - `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_multi_core_w_analysis.md`
  - `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/data_movement/untilize/device/factories/untilize_multi_core_analysis.md`
- DeepWiki queries:
  - "How does reduce_tile work for variance computation along W dimension?" - Found reduce_init/reduce_tile pattern, REDUCE_ROW for W reduction
  - "How does generate_reduce_scaler work?" - Scaler is packed bfloat16 in uint32, fills 512 uint32 entries
  - "How does sub_tiles work?" - sub_tiles_bcast_scalar for scalar broadcast subtraction
  - "How does mul_tiles/square_tile work?" - square_tile is SFPU primitive for element-wise squaring
  - "How do tilize/untilize helpers work?" - tilize_init/tilize_block/tilize_uninit pattern
- Documentation consulted:
  - METALIUM_GUIDE.md for Tensix architecture
  - compute_kernel_api/tilize.h, untilize.h for helper function signatures
  - reduce_helpers.hpp for reduce library patterns
