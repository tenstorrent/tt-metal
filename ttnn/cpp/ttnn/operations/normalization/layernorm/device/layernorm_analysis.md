# LayerNorm Implementation Analysis

## Overview

LayerNorm is a normalization operation that normalizes input tensors across the last dimension (feature dimension). The implementation supports both standard LayerNorm and RMSNorm variants, with optional fused operations including:
- Pre-addition of a residual tensor (x = a + b)
- Gamma scaling (multiplication)
- Beta offset (addition)

The mathematical formula is:
- **LayerNorm**: `output = (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta`
- **RMSNorm**: `output = x / sqrt(E[x^2] + eps) * gamma + beta` (skips mean subtraction)

### Program Factory Files
- **Interleaved**: `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/normalization/layernorm/device/layernorm_op_multi_core.cpp`
- **Sharded**: `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/normalization/layernorm/device/layernorm_op_multi_core_sharded.cpp`

## Work Unit Definition

| Attribute | Interleaved | Sharded |
|-----------|-------------|---------|
| **Granularity** | Tile row | Tile block (block_ht x block_wt tiles) |
| **Unit size** | Wt tiles (one row of tiles along width) | block_ht * block_wt tiles |
| **Total units** | NC * Ht tile rows | One block per core |
| **Loop structure** | Outer: tile rows; Inner: blocks of Wt | Single pass per core with subblock iterations |

For interleaved mode, each core processes `num_tile_rows_per_core` complete tile rows. For sharded mode, each core processes its assigned shard (block_ht x block_wt tiles).

## Tensor Format and Layout

### Input Tensor(s)

| Property | Input (a) | Residual (b) | Gamma | Beta |
|----------|-----------|--------------|-------|------|
| **Logical shape** | [..., W] | [..., W] | [1, 1, 1, W] | [1, 1, 1, W] |
| **Dimension convention** | NHWC (last dim normalized) | NHWC | 1D broadcast | 1D broadcast |
| **Tensor layout** | TILE_LAYOUT | TILE_LAYOUT | TILE_LAYOUT or ROW_MAJOR | TILE_LAYOUT or ROW_MAJOR |
| **Memory layout** | INTERLEAVED or SHARDED | INTERLEAVED or SHARDED | INTERLEAVED | INTERLEAVED |
| **Buffer type** | DRAM or L1 | DRAM or L1 | DRAM | DRAM |
| **Data type** | BFLOAT16/FLOAT32 | BFLOAT16/FLOAT32 | BFLOAT16/FLOAT32 | BFLOAT16/FLOAT32 |

### Output Tensor

| Property | Output |
|----------|--------|
| **Logical shape** | [..., W] (same as input) |
| **Dimension convention** | NHWC |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM or L1 |
| **Data type** | BFLOAT16/FLOAT32 |

### Sharded Configuration (when using sharded mode)

| Property | Value |
|----------|-------|
| **Shard Shape** | [block_h, block_w] elements = [block_ht * 32, block_wt * 32] |
| **Core Grid** | Derived from shard_spec.grid |
| **Shard Orientation** | ROW_MAJOR or COL_MAJOR |
| **Memory Layout** | HEIGHT_SHARDED, WIDTH_SHARDED, or BLOCK_SHARDED |

### Layout Transformations

1. **Tile transpose**: Welford's algorithm requires transposing input tiles to process columns as rows
2. **Mean/Variance transpose**: Results are transposed back to column format for broadcasting
3. **Resharding (post-allgather)**: Output may be resharded to different cores if output shard spec differs from input

## Data Flow Pattern

### Interleaved Mode Data Flow

```
Stage 1: Reader reads input tiles from DRAM -> cb_in0
         Reader reads residual tiles from DRAM -> cb_in1 (if FUSE_PRE_ADD)
         Reader generates scaler tile -> cb_scaler (for reduction)
         Reader generates epsilon tile -> cb_eps

Stage 2: Compute performs:
         a) Pre-add: cb_in0 + cb_in1 -> cb_x (if FUSE_PRE_ADD)
         b) Mean reduction: cb_x -> cb_ex (E[x])
         c) Subtract mean: cb_x - cb_ex -> cb_xmm
         d) Square: cb_xmm^2 -> cb_xmm2
         e) Variance reduction: cb_xmm2 -> cb_ex2 (Var[x])
         f) Add epsilon and rsqrt: sqrt(cb_ex2 + eps)^-1 -> cb_ex2pe
         g) Normalize: cb_xmm * cb_ex2pe -> cb_fusion
         h) Apply gamma: cb_fusion * cb_gamma -> cb_fusion (if do_gamma)
         i) Apply beta: cb_fusion + cb_beta -> cb_out (if do_beta)

Stage 3: Writer writes cb_out -> DRAM
```

### Sharded Mode Data Flow

```
Stage 1: Input data already in L1 (cb_in0 aliased to input buffer)
         Writer reads gamma/beta from DRAM -> cb_gamma, cb_beta
         Writer generates scaler tiles

Stage 2: Compute performs partial reduction:
         a) Pre-add (if enabled): cb_in0 + cb_in1 -> cb_x
         b) Partial mean/var: Welford algorithm or reduction -> cb_ex_partial

Stage 3: Reader coordinates multicast:
         a) Sender waits for all cores' partials
         b) Sender reads partials from all cores -> cb_ex_external
         c) Compute combines partials -> cb_ex
         d) Sender multicasts global results -> cb_ex_global (all cores)

Stage 4: Compute completes normalization:
         a) Subtract mean: cb_in - cb_ex_global[mean] -> cb_xmm
         b) Scale: cb_xmm * cb_ex_global[1/sqrt(var+eps)] -> cb_im
         c) Apply gamma/beta -> cb_out

Stage 5: Writer handles resharding if needed
```

## Circular Buffer Configuration

### Interleaved Mode Circular Buffers

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_in0 | Input tensor a | Wt tiles (or 2*blk if large) | blk tiles | Single/Multi | Reader | Compute | Block |
| c_1 | cb_in1 | Residual tensor b | 2*blk tiles | blk tiles | Double | Reader | Compute | Block |
| c_2 | cb_scaler | Reduction scaler (1/W) | 2 tiles | 1 tile | Single | Reader | Compute | Program |
| c_3 | cb_eps | Epsilon value | 2 tiles | 1 tile | Single | Reader | Compute | Program |
| c_5 | cb_gamma | Gamma weights | Wt tiles | blk tiles | Single | Reader | Compute | Program |
| c_6 | cb_beta | Beta weights | Wt tiles | blk tiles | Single | Reader | Compute | Program |
| c_16 | cb_out | Output tensor | 2*blk tiles | blk tiles | Double | Compute | Writer | Block |
| c_18 | cb_ex | E[x] mean | 2 tiles | 1 tile | Single | Compute | Compute | Block |
| c_19 | cb_ex2 | Var[x] | 2 tiles | 1 tile | Single | Compute | Compute | Block |
| c_20 | cb_xmm2 | (x-E[x])^2 | Wt tiles | blk tiles | Single | Compute | Compute | Block |
| c_21 | cb_ex2pe | 1/sqrt(Var+eps) | 8 tiles | 1 tile | Single | Compute | Compute | Block |
| c_22 | cb_fusion | Gamma/beta intermediate | 2*blk tiles | blk tiles | Double | Compute | Compute | Block |
| c_23 | cb_x | x=a+b (pre-add result) | Wt tiles | blk tiles | Single | Compute | Compute | Block |
| c_24 | cb_xmm | x - E[x] | Wt tiles | blk tiles | Single | Compute | Compute | Block |
| c_25 | cb_reciprocals | Welford reciprocal LUT | W floats | N/A | Single | Reader | Compute | Program |

### Sharded Mode Circular Buffers

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_in0 | Input (aliased to sharded buffer) | block_ht*block_wt tiles | N/A | Single | Sharded | Compute | Program |
| c_1 | cb_in1 | Residual (aliased to sharded buffer) | block_ht*block_wt tiles | N/A | Single | Sharded | Compute | Program |
| c_2 | cb_scaler | Row reduction scaler | 1 tile | 1 tile | Single | Writer | Compute | Program |
| c_3 | cb_eps | Epsilon value | 1 tile | 1 tile | Single | Writer | Compute | Program |
| c_4 | cb_scaler_global | Global reduction scaler | 1 tile | 1 tile | Single | Writer | Compute | Program |
| c_5 | cb_gamma | Gamma weights | block_wt tiles | block_wt tiles | Single | Writer | Compute | Program |
| c_6 | cb_beta | Beta weights | block_wt tiles | block_wt tiles | Single | Writer | Compute | Program |
| c_8 | cb_ex_partial | Partial E[x] (local) | block_ht tiles | 1 tile | Single | Compute | Reader | Block |
| c_9 | cb_ex | Combined E[x] | block_ht tiles | 1 tile | Single | Compute | Reader | Block |
| c_10 | cb_ex_external | External partials | num_blocks tiles | 1 tile | Single | Reader | Compute | Block |
| c_11 | cb_ex_partial2 | Partial Var[x] | block_ht tiles | 1 tile | Single | Compute | Reader | Block |
| c_12 | cb_ex2 | Combined Var[x] | block_ht tiles | 1 tile | Single | Compute | Reader | Block |
| c_13 | cb_ex_external2 | External variance partials | num_blocks tiles | 1 tile | Single | Reader | Compute | Block |
| c_15 | cb_ex_global | Global mean/var (multicasted) | block_ht tiles | 1 tile | Single | Reader | Compute | Block |
| c_16 | cb_out | Output (may alias sharded buffer) | block_ht*block_wt tiles | N/A | Single | Compute | Writer | Program |
| c_17 | cb_out_resharded | Resharded output | varies | varies | Single | Writer | External | Program |
| c_18 | cb_xmm | x - E[x] | block_ht*block_wt tiles | subblock tiles | Single | Compute | Compute | Block |
| c_20 | cb_ex2pe | 1/sqrt(Var+eps) | block_ht tiles | 1 tile | Single | Compute | Compute | Block |
| c_22 | cb_transpose | Welford transpose workaround | 2*block_ht tiles | varies | Single | Compute | Compute | Block |
| c_24 | cb_x | Pre-add result / intermediate | block_ht*block_wt tiles | subblock tiles | Single | Compute | Compute | Block |
| c_25 | cb_reciprocals | Welford reciprocal LUT | W floats | N/A | Single | Sharded | Compute | Program |

## Pipeline Pattern Summary

### Interleaved Mode
- **Input/Output CBs (c_0, c_1, c_16)**: Double-buffered for overlap between reader/writer and compute
- **Intermediate CBs (c_24, c_18, c_20)**: Single-buffered, capacity matches row width (Wt tiles)
- **Scalar CBs (c_2, c_3)**: Generated once, reused across all rows

### Sharded Mode
- **Input CBs (c_0, c_1)**: Globally allocated (aliased to sharded tensor buffer)
- **Partial reduction CBs**: Single-buffered, used for cross-core communication
- **Output CB**: May be globally allocated or used for resharding writes

## Index Calculations

### Interleaved Mode
```cpp
// Tile offset calculation for each core
uint32_t tile_offset = curr_row * Wt;  // Starting tile for this core
uint32_t tile_id = tile_offset + block.start() + w;  // Tile within current row

// Physical to logical mapping
uint32_t NC = a.physical_volume() / (Hp * Wp);  // Batch * Channel
uint32_t Wt = Wp / TILE_WIDTH;  // Tiles in width
uint32_t Ht = Hp / TILE_HEIGHT;  // Tiles in height
uint32_t num_tile_rows = NC * Ht;  // Total tile rows to process
```

### Sharded Mode
```cpp
// Core position within grid
uint32_t height_index = mcast_1d ? 0 : (row_wise ? core.y : core.x);
uint32_t width_index = mcast_1d ? i : (row_wise ? core.x : core.y);

// Tile offset within partial buffer
uint32_t all_to_all_worker_tile_offset = width_index * num_rows_per_all_to_all_worker;

// Gamma/beta tile start
uint32_t gamma_tile_start_id = width_index * block_wt;
```

## Memory Access Patterns

### Read Pattern

**Interleaved Mode:**
- Sequential tile reads within each row (tiles 0 to Wt-1)
- Row-by-row progression through assigned tile rows
- Gamma/beta read once (first row only) and cached

**Sharded Mode:**
- Input data already in L1 (no DRAM reads for input)
- Gamma/beta read from DRAM by writer kernel
- Partial results read from remote L1 via NOC unicast reads
- Global results received via NOC multicast

### Write Pattern

**Interleaved Mode:**
- Sequential tile writes within each row
- Row-by-row progression through output buffer
- Blocked writes (blk tiles at a time)

**Sharded Mode:**
- Output written to sharded buffer in L1
- Optional resharding: tiles written to remote cores via NOC writes
- Reshard pattern: segment-by-segment to potentially different storage cores

## Core Distribution Strategy

### Interleaved Mode

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (linearized) |
| **Grid dimensions** | device->compute_with_storage_grid_size() |
| **Total cores** | num_cores (from split_work_to_cores) |
| **Work per core** | num_tile_rows_per_core_group_1 or num_tile_rows_per_core_group_2 |
| **Load balancing** | Two-group split (some cores get +1 row) |
| **Remainder handling** | core_group_2 handles remainder rows |

### Sharded Mode

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 2D (from shard_spec.grid) |
| **Grid dimensions** | grid_size.x x grid_size.y |
| **Total cores** | shard_spec.num_cores() |
| **Work per core** | block_ht x block_wt tiles (one shard) |
| **Load balancing** | Equal (each core has same shard size) |
| **Special roles** | Sender core (0,0), all-to-all workers, non-all-to-all workers |

#### Core Role Classification (Sharded)

1. **Sender Core**: First core (0,0) - coordinates multicast, collects and distributes global results
2. **All-to-all Workers**: Cores that participate in the reduction combine (first `num_all_to_all_workers` cores)
3. **Non-all-to-all Workers**: Remaining cores that only receive multicasted results
4. **Second-stage Readers** (two-stage reduce): Top row of cores that combine first-stage results

## Arguments

### Compile-Time Arguments (Interleaved Reader)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | block_size | uint32_t | Number of tiles per block (blk) |
| 1 | use_welford | uint32_t | Whether to use Welford's algorithm |
| 2-N | tensor_accessor_args | TensorAccessorArgs | Access patterns for a, b, gamma, beta |

### Runtime Arguments (Interleaved Reader)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | a_addr | uint32_t | Input tensor buffer address |
| 1 | num_tile_rows_per_core | uint32_t | Tile rows assigned to this core |
| 2 | Wt | uint32_t | Tiles in width dimension |
| 3 | tile_offset | uint32_t | Starting tile offset |
| 4 | packed_one_value | uint32_t | Packed bfloat16 value of 1.0 |
| 5 | eps | uint32_t | Epsilon as bit-cast float |
| 6 | gamma_dram_addr | uint32_t | Gamma buffer address |
| 7 | beta_dram_addr | uint32_t | Beta buffer address |
| 8 | b_dram_addr | uint32_t | Residual tensor address |
| 9 | W | uint32_t | Logical width (for partial tile handling) |

### Compile-Time Arguments (Sharded Sender Reader)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | reduce_receiver_semaphore_id | uint32_t | Semaphore for receiver sync |
| 1 | reduce_sender_semaphore_id | uint32_t | Semaphore for sender sync |
| 2 | num_blocks | uint32_t | Total number of blocks (cores) |
| 3 | block_h | uint32_t | Block height in tiles |
| 4 | block_h_size_bytes | uint32_t | Block height in bytes |
| 5 | num_all_to_all_workers_first_stage | uint32_t | Workers in first reduce stage |
| 6 | num_tiles_per_worker | uint32_t | Tiles per worker |
| 7 | num_tiles_per_worker_bytes | uint32_t | Tiles per worker in bytes |
| 8 | num_tiles_per_worker_last | uint32_t | Tiles for last worker |
| 9 | num_tiles_per_worker_last_bytes | uint32_t | Last worker tiles in bytes |
| 10 | row_major | uint32_t | Whether orientation is row-major |
| 11 | num_x | uint32_t | Grid width |
| 12 | num_y | uint32_t | Grid height |
| 13 | use_two_stage_reduce | uint32_t | Whether using two-stage reduction |
| 14 | num_blocks_first_stage | uint32_t | Blocks in first stage |
| 15 | num_blocks_second_stage | uint32_t | Blocks in second stage |
| 16 | reduce_second_stage_semaphore_id | uint32_t | Second stage sync semaphore |
| 17 | rms_norm | uint32_t | Whether doing RMSNorm |
| 18 | use_welford | uint32_t | Whether using Welford's algorithm |

### Runtime Arguments (Sharded Sender Reader)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | mcast_dest_noc_start_x | uint32_t | Multicast destination start X |
| 1 | mcast_dest_noc_start_y | uint32_t | Multicast destination start Y |
| 2 | mcast_dest_noc_end_x | uint32_t | Multicast destination end X |
| 3 | mcast_dest_noc_end_y | uint32_t | Multicast destination end Y |
| 4 | start_x | uint32_t | This core's X position |
| 5 | start_y | uint32_t | This core's Y position |
| 6+ | in0_remote_noc_x[] | uint32_t[] | NOC X coordinates of all cores |
| 6+num_x+ | in0_remote_noc_y[] | uint32_t[] | NOC Y coordinates of all cores |

## Kernel Implementations

### Interleaved Mode Kernels

#### Reader Kernel
- **File**: `ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/reader_unary_interleaved_ln.cpp`
- **Core**: RISCV_0
- **NOC**: NOC0 (preferred for DRAM read)

| Input | Output | Operations |
|-------|--------|------------|
| DRAM (a, b, gamma, beta) | cb_in0, cb_in1, cb_gamma, cb_beta | noc_async_read_tile |

**Key Logic**:
- Generates reduction scaler tile (1/W packed as bfloat16)
- Generates epsilon tile for variance calculation
- Reads input tiles in blocks
- Conditionally reads residual, gamma, beta based on defines

#### Compute Kernel
- **File**: `ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/layernorm.cpp`
- **Core**: RISCV_2 (Tensix compute)

| Input | Output | Operations |
|-------|--------|------------|
| cb_in0, cb_in1, cb_scaler, cb_eps, cb_gamma, cb_beta | cb_out | add_tiles, reduce_tile, sub_tiles_bcast_cols, mul_tiles, rsqrt_tile, mul_tiles_bcast_rows, add_tiles_bcast_rows |

**Key Logic**:
1. Optional pre-add: `x = a + b`
2. Mean calculation: `E[x] = reduce_row(x) * scaler`
3. Mean subtraction: `xmm = x - E[x]` (broadcast column)
4. Variance: `E[xmm^2] = reduce_row(xmm * xmm) * scaler`
5. Normalization: `rsqrt(var + eps)`, then `xmm * rsqrt_result`
6. Optional gamma: `result * gamma` (broadcast row)
7. Optional beta: `result + beta` (broadcast row)

#### Writer Kernel
- **File**: `ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/writer_unary_interleaved_start_id_blocked.cpp`
- **Core**: RISCV_1
- **NOC**: NOC1 (preferred for DRAM write)

| Input | Output | Operations |
|-------|--------|------------|
| cb_out | DRAM | noc_async_write_tile |

**Key Logic**:
- Writes output tiles in blocks
- Maintains tile_id for sequential DRAM addressing

### Sharded Mode Kernels

#### Sender Reader Kernel
- **File**: `ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/reader_mcast_sender_unary_sharded_ln.cpp`
- **Core**: RISCV_0
- **NOC**: NOC0

| Input | Output | Operations |
|-------|--------|------------|
| cb_ex_partial (local), remote L1 | cb_ex_external, cb_ex_global | noc_async_read, noc_async_write_multicast, semaphore operations |

**Key Logic**:
1. Wait for local partial results
2. Signal other cores that partials are ready
3. Read partials from all cores in row/column
4. Wait for combined results
5. Collect all combined results
6. Multicast global results to all cores

#### Receiver Reader Kernel
- **File**: `ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/reader_mcast_receiver_unary_sharded_ln.cpp`
- **Core**: RISCV_0
- **NOC**: NOC0

| Input | Output | Operations |
|-------|--------|------------|
| cb_ex_partial (local), remote L1 | cb_ex_external, cb_ex_global | noc_async_read, semaphore operations |

**Key Logic**:
1. Wait for local partial results
2. Signal sender that partials are ready
3. Wait for sender's "go" signal
4. Read partials from other cores (if all-to-all worker)
5. Signal sender when combine is done
6. Receive multicasted global results

#### Sharded Compute Kernel
- **File**: `ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/layernorm_sharded.cpp`
- **Core**: RISCV_2 (Tensix compute)

| Input | Output | Operations |
|-------|--------|------------|
| cb_in0, cb_in1, cb_ex_external, cb_ex_global, cb_gamma, cb_beta | cb_ex_partial, cb_ex, cb_out | reduce_tile, add_tiles, sub_tiles_bcast_cols, mul_tiles_bcast_cols/rows, rsqrt_tile |

**Key Logic**:
1. Optional pre-add
2. Partial reduction for E[x] and Var[x]
3. Global reduction (combine partials from cb_ex_external)
4. Compute 1/sqrt(var + eps)
5. Normalize using global E[x] and rsqrt
6. Apply gamma/beta

#### Writer Kernel (Sharded)
- **File**: `ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/writer_unary_sharded_ln.cpp`
- **Core**: RISCV_1
- **NOC**: NOC1

| Input | Output | Operations |
|-------|--------|------------|
| DRAM (gamma, beta) | cb_gamma, cb_beta, cb_scaler, cb_eps | noc_async_read_tile, generate_reduce_scaler |

**Key Logic**:
- Generates scalar tiles for reduction
- Reads gamma and beta from DRAM
- Handles resharding writes if output shard spec differs

### Welford Variant Kernels

#### Welford Compute Kernel
- **File**: `ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/layernorm_welford.cpp`

**Key Differences from Standard**:
- Uses `welford_init()`, `welford_update()`, `welford_finalize_to_row()` instead of separate mean/variance passes
- Requires transpose of input tiles (`transpose_wh_tile`)
- Uses pre-computed reciprocal LUT for numerical stability
- Computes mean and variance simultaneously in single pass

#### Sharded Welford Compute Kernel
- **File**: `ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/layernorm_sharded_welford.cpp`

**Key Differences**:
- Uses `combine_welford_partials()` for merging partial statistics from multiple cores
- Supports both single-stage and two-stage reduction
- Handles edge cases for partial tiles at tensor boundaries

## Multicast Pattern (Sharded Mode)

### Single-Stage Reduction

1. Each core computes partial mean/variance for its shard
2. Sender core waits for all partials via semaphore
3. Sender signals all cores to start reading remote partials
4. Each all-to-all worker reads partials from cores in same row
5. Each worker combines partials into local result
6. Workers signal sender when combine is done
7. Sender collects all combined results from row
8. Sender multicasts global results to all cores in grid

### Two-Stage Reduction (Width-Sharded)

1. **First Stage**: Each core row performs single-stage reduction
2. First-stage results stored in cb_ex (not cb_ex_global)
3. **Second Stage**: Top row cores ("second-stage readers") receive first-stage results from their column
4. Second-stage readers combine all first-stage results
5. Sender multicasts final global results to all cores

### Semaphore Usage

| Semaphore | Purpose |
|-----------|---------|
| reduce_receiver_semaphore | Receivers signal sender when partials ready |
| reduce_sender_semaphore | Sender signals receivers to proceed |
| reduce_second_stage_semaphore | Coordinates two-stage reduction |

## Implementation Notes

### Algorithm Variants

1. **Standard Reduction**: Two-pass algorithm computing mean then variance
2. **Welford's Algorithm**: Single-pass numerically stable algorithm
   - Better precision for large tensors or small variance values
   - Requires reciprocal LUT for efficiency
   - Uses tile transpose for row-wise processing

### Large Tensor Handling (Interleaved)

When circular buffers cannot fit in L1:
- Uses `large_tensor_needed` flag
- Reduces buffer sizes (Wt_next_block_up = 56 or 112)
- Uses specialized kernel: `layernorm_large_tensor.cpp`
- Processes tensor in multiple passes

### FP32 Accumulation

When `fp32_dest_acc_en = true`:
- Block size reduced from 8 to 4 (due to dest register constraints)
- Intermediate CB format becomes Float32
- More accurate for large reductions

### RMSNorm Optimization

When `rms_norm = true`:
- Skips mean calculation (E[x] step)
- cb_xmm can alias cb_in (no separate storage needed)
- Fewer circular buffers required

### Subblock Processing (Sharded)

- Uses `subblock_wt` for partial tile processing
- Volatile loop bounds prevent excessive unrolling
- Enables processing of wide tensors without code size explosion

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the sharded memory layout work in tt-metal?"
   **Reason**: Needed to understand ShardSpec, shard orientation, and data distribution
   **Key Findings**: ShardSpec contains grid, shape, orientation. HEIGHT_SHARDED divides along height, WIDTH_SHARDED along width. Row-major vs column-major affects core traversal order.

2. **Query**: "What is the multicast pattern in tt-metal?"
   **Reason**: Needed to understand noc_async_write_multicast and semaphore coordination
   **Key Findings**: Multicast sends data from one core to rectangular grid of cores. Semaphores used for synchronization - wait/set/inc operations coordinate sender/receivers.

3. **Query**: "What is Welford's algorithm and how is it used in tt-metal?"
   **Reason**: Needed to understand alternative mean/variance calculation approach
   **Key Findings**: Welford's computes mean and variance in single pass with better numerical stability. Uses incremental updates with reciprocal values. Partial results can be combined across distributed computation.

### Documentation References

1. **Source**: `/localdev/mstaletovic/tt-metal/.claude/CLAUDE.md`
   **Reason**: General codebase understanding
   **Key Information**: Tensix cores use 32x32 tiles, kernel types (Reader/Compute/Writer), NoC handles data movement

2. **Source**: Program factory source files
   **Reason**: Primary implementation analysis
   **Key Information**: Complete CB setup, kernel selection logic, runtime argument construction, work distribution algorithms
