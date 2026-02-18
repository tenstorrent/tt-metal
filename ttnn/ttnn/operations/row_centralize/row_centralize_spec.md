# Row Centralize Functional Specification

## Overview
- **Operation Name**: row_centralize
- **Category**: normalization
- **Planning Mode**: Hybrid
- **Reference Operations**: tilize, untilize, batch_norm
- **Reference Analyses**:
  - `ttnn/ttnn/operations/row_centralize/agent_logs/tilize_analysis.md` (role: input_stage)
  - `ttnn/ttnn/operations/row_centralize/agent_logs/untilize_analysis.md` (role: output_stage)
  - `ttnn/ttnn/operations/row_centralize/agent_logs/batch_norm_analysis.md` (role: compute_core)
- **Workflow**: Generic Op (Python-based)

## Mathematical Definition

### Formula
```
For each row r in the input tensor:

  mu_r    = (1 / W) * sum(x[r, 0..W-1])         # row mean
  c_r[j]  = x[r, j] - mu_r                       # centralize
  var_r   = (1 / W) * sum(c_r[j]^2, j=0..W-1)   # row variance
  s_r     = rsqrt(var_r + epsilon)                # inverse std
  y[r, j] = c_r[j] * s_r                         # standardize
```

Equivalently, for a tensor `x` of shape `[..., W]`:
```
mu      = mean(x, dim=-1, keepdim=True)
c       = x - mu
var     = mean(c^2, dim=-1, keepdim=True)
s       = rsqrt(var + epsilon)
y       = c * s
```

### Semantic Description
Row-wise standardization (also known as instance standardization along the last dimension). For each row of the input tensor, compute the mean, subtract it, compute the variance of the centered values, then divide by the standard deviation (plus epsilon for numerical stability). The output has zero mean and unit variance along each row.

This is mathematically equivalent to LayerNorm without learnable affine parameters (gamma=1, beta=0) applied to the last dimension.

## API Specification

### Parameters
| Parameter | Type | Required | Valid Range | Default | Description |
|-----------|------|----------|-------------|---------|-------------|
| input_tensor | Tensor | Yes | At least 2D, bfloat16, row-major, interleaved, on device | - | Input tensor to standardize |
| epsilon | float | No | (0, inf) | 1e-5 | Small constant for numerical stability in rsqrt |

### Input Tensor Requirements

| Property | Requirement | Error Message Hint |
|----------|-------------|-------------------|
| Rank | >= 2 | "Input tensor must have rank >= 2" |
| Layout | ROW_MAJOR | "Input tensor must be in ROW_MAJOR layout" |
| Dtype | BFLOAT16 | "Input tensor dtype must be bfloat16" |
| Memory layout | INTERLEAVED | "Input tensor must have interleaved memory layout" |
| Device | Must be on device | "Input tensor must be on device" |
| Last dim | Must be divisible by 32 (TILE_WIDTH) | "Input tensor last dimension must be divisible by 32" |
| 2nd-to-last dim | Must be divisible by 32 (TILE_HEIGHT) | "Input tensor second-to-last dimension must be divisible by 32" |

Note on padding: Requiring divisibility by 32 avoids the complexity of padding/unpadding in this initial single-core implementation. A future version can relax this by padding the input before tilize.

### Output Tensor Specification
- **Shape**: Same as input (no dimension change)
- **Dtype**: BFLOAT16
- **Layout**: ROW_MAJOR
- **Memory layout**: INTERLEAVED
- **Buffer type**: DRAM

## Component Sources (Hybrid Mode)

This operation is composed from multiple references.

### Input Stage (from tilize)
| Component | Source | Modifications |
|-----------|--------|---------------|
| Reader kernel | tilize `reader_unary_stick_layout_split_rows_interleaved` | Single-core only (no multi-core split). Additionally generates reduce scaler tile and epsilon scalar tile at startup. |
| CB c_0 (cb_rm_in) | tilize CB c_0 | Same sizing: Wt tiles (one tile-row of RM sticks). Page size = tile_size for BF16. |
| Compute (tilize phase) | tilize compute `tilize_block` via `compute_kernel_lib::tilize` | Reused directly. Tilize each block of Wt tiles from c_0 to c_1. |

### Compute Stage (new, inspired by batch_norm)
| Component | Source | Modifications |
|-----------|--------|---------------|
| Reduce mean (REDUCE_ROW) | kernel_lib `reduce_helpers_compute.hpp` | New. Reduce row-wise SUM with 1/W scaler to get mean. Uses `compute_kernel_lib::reduce<SUM, REDUCE_ROW>`. |
| Subtract mean (COL broadcast) | kernel_lib `binary_op_helpers.hpp` | New. `compute_kernel_lib::sub<COL>` to subtract mean from each tile in the row. |
| Square centered values | kernel_lib `binary_op_helpers.hpp` | New. `compute_kernel_lib::square` on centered tiles. |
| Reduce variance (REDUCE_ROW) | kernel_lib `reduce_helpers_compute.hpp` | New. Reduce row-wise SUM with 1/W scaler on squared values to get variance. |
| Add epsilon (SCALAR broadcast) | kernel_lib `binary_op_helpers.hpp` | Inspired by batch_norm epsilon addition. `compute_kernel_lib::add<SCALAR>` to add epsilon to variance tile. |
| Rsqrt | Compute API `rsqrt_tile` | Inspired by batch_norm rsqrt pattern. Applied to (variance + epsilon) tile in DST. |
| Multiply by inv_std (COL broadcast) | kernel_lib `binary_op_helpers.hpp` | New. `compute_kernel_lib::mul<COL>` to multiply centered tiles by rsqrt result. |

### Output Stage (from untilize)
| Component | Source | Modifications |
|-----------|--------|---------------|
| Compute (untilize phase) | untilize compute via `compute_kernel_lib::untilize` | Reused directly. Untilize each block of Wt tiles from cb_result to cb_rm_out. |
| CB cb_rm_out | untilize CB c_16 | Same pattern: Wt tiles, page size = tile_size for BF16. |
| Writer kernel | untilize `writer_unary_stick_layout_split_rows_multi_core` | Adapted for single-core. Writes RM sticks from cb_rm_out to DRAM via TensorAccessor. |

### Interface Compatibility
| Interface | Component A | Component B | Format A | Format B | Compatible? |
|-----------|------------|-------------|----------|----------|-------------|
| Reader -> Compute (tilize) | RM stick reader -> CB c_0 | tilize reads CB c_0 | RM sticks, Wt tiles worth of data per block | RM sticks (tilize expects RM input) | Yes |
| Compute (tilize) -> Compute (reduce/sub/mul) | tilize writes CB c_1 | reduce/sub/mul read CB c_1 | TILE format, Wt tiles per block | TILE format | Yes |
| Compute (standardize) -> Compute (untilize) | mul writes CB c_5 | untilize reads CB c_5 | TILE format, Wt tiles per block | TILE format (untilize expects tile input) | Yes |
| Compute (untilize) -> Writer | untilize writes CB c_16 | Writer reads CB c_16 | RM format, Wt tiles per block | RM sticks | Yes |

### CB ID Resolution
| Logical CB | Source Ref | Original ID | Final ID | Notes |
|------------|-----------|-------------|----------|-------|
| cb_rm_in (RM sticks from reader) | tilize | c_0 | c_0 | Input staging for RM sticks from DRAM |
| cb_tilized (tilized tiles) | tilize output / compute input | c_16 | c_1 | Tilize output, compute input. Moved from c_16 to c_1 to free c_16 for RM output |
| cb_mean (row mean tile) | new | - | c_2 | Stores mean tile from reduce_row |
| cb_centered (centered tiles) | new | - | c_3 | x - mean result |
| cb_squared (squared tiles) | new | - | c_24 | c^2 intermediate |
| cb_var (variance tile) | new | - | c_4 | Variance tile from reduce_row |
| cb_inv_std (rsqrt result) | new | - | c_5 | 1/sqrt(var + eps) |
| cb_result (standardized tiles) | new | - | c_6 | Final standardized tiles in TILE format |
| cb_eps (epsilon scalar) | batch_norm | c_4 | c_7 | Epsilon constant tile |
| cb_scaler (reduce scaler) | new | - | c_8 | 1/W reduce scaler tile |
| cb_var_plus_eps (var + eps) | new | - | c_25 | Intermediate for var + epsilon before rsqrt |
| cb_rm_out (RM sticks for writer) | untilize | c_16 | c_16 | Output staging for RM sticks to DRAM |

## Design Decisions

### Decision 1: Single-Core Execution
- **Choice**: Run everything on a single Tensix core.
- **Rationale**: Simplifies the initial implementation. Row-wise standardization requires full-row data (all Wt tiles in a row) for the mean/variance reduction, meaning work cannot trivially be split across the width dimension. Multi-core would require splitting across tile-rows (height dimension), which adds complexity without changing the core algorithm.
- **Alternatives Considered**: Multi-core with height-based parallelization (each core handles a subset of tile-rows). This is a natural future optimization.
- **Tradeoffs**: Lower throughput for large tensors, but correct and debuggable. Multi-core can be added later by wrapping the single-core logic with work distribution.

### Decision 2: Block-at-a-Time Processing (One Tile-Row per Iteration)
- **Choice**: Process one tile-row (Ht=1, Wt tiles) at a time through the full pipeline: tilize -> reduce -> sub -> square -> reduce -> add_eps -> rsqrt -> mul -> untilize.
- **Rationale**: This minimizes peak L1 memory usage. Each tile-row requires ~Wt tiles of CB space for the main data plus a few single-tile CBs for statistics. Processing the full tensor at once would require all tiles in L1 simultaneously.
- **Alternatives Considered**: (a) Process entire tensor at once (too much L1). (b) Process multiple tile-rows (more complex CB management). (c) Fused approach where tilize/untilize are done incrementally within the compute phase.
- **Tradeoffs**: Good L1 efficiency at the cost of some kernel reconfiguration overhead per tile-row (tilize_init/uninit, reduce_init/uninit, untilize_init/uninit).

### Decision 3: Reuse Centered Tiles for Squaring (Two Passes Over Centered Data)
- **Choice**: After computing `c = x - mean`, the centered tiles in CB c_3 are used for both: (a) squaring to compute variance, and (b) the final multiplication by inv_std. This requires keeping the centered tiles available across both uses.
- **Rationale**: Avoids recomputing centered values. The centered tiles are produced once by `sub<COL>`, then: first pass reads them with `WaitUpfrontNoPop` for squaring, second pass reads them with `NoWaitNoPop` for the final multiply. The centered tiles remain in the CB across both operations.
- **Alternatives Considered**: (a) Recompute centered values (wasteful). (b) Store centered values in a separate buffer and copy (extra CB, extra memory).
- **Tradeoffs**: Requires careful CB synchronization -- the centered CB must not be popped until both consumers are done.

### Decision 4: Compute Kernel Performs All Stages (Tilize + Math + Untilize)
- **Choice**: A single compute kernel handles tilize, all normalization math, and untilize.
- **Rationale**: The tilize and untilize operations happen on the compute RISC-V (they configure the unpacker/packer). Having a single compute kernel avoids the overhead of multiple kernel launches and allows tight integration of the compute pipeline. The kernel helper libraries (`tilize_helpers.hpp`, `untilize_helpers.hpp`, `reduce_helpers_compute.hpp`, `binary_op_helpers.hpp`) support this chained pattern with init/uninit mode controls.
- **Alternatives Considered**: Separate compute kernels for tilize, math, and untilize (would require additional CB synchronization and multiple kernel launches).
- **Tradeoffs**: Longer, more complex compute kernel but better performance and simpler CB management.

### Decision 5: Use Kernel Helper Libraries
- **Choice**: Use `compute_kernel_lib::tilize`, `compute_kernel_lib::untilize`, `compute_kernel_lib::reduce`, `compute_kernel_lib::sub`, `compute_kernel_lib::mul`, `compute_kernel_lib::square`, and `compute_kernel_lib::add` from the kernel_lib directory.
- **Rationale**: These libraries encapsulate correct CB management, DST register handling, init/uninit patterns, and data format reconfiguration. They are production-quality, tested, and handle edge cases.
- **Alternatives Considered**: Raw LLK calls (error-prone, verbose). Writing custom compute loops (reinventing the wheel).
- **Tradeoffs**: Some overhead from generality of the helpers, but correctness and maintainability are prioritized.

### Decision 6: Epsilon and Scaler Generation in Reader
- **Choice**: The reader kernel generates two special tiles at startup: (a) the reduce scaler tile (1/W packed as bf16) using `generate_reduce_scaler`, and (b) the epsilon scalar tile using `generate_bcast_scalar_bfloat16`. Both are pushed once and persist for the entire program.
- **Rationale**: Follows the batch_norm pattern where the reader fills the epsilon CB at startup. The scaler tile is needed by the reduce helper and must be in its CB before reduce is called.
- **Alternatives Considered**: Generating these tiles on the host and passing them as input tensors (overkill for constants). Generating them in the compute kernel (compute does not have direct L1 write access for arbitrary CB filling).
- **Tradeoffs**: Reader has a small one-time startup cost for tile generation.

### Decision 7: Reduce Scaler Packing Convention
- **Choice**: The reduce scaler value `1/W` is packed as `(bf16 << 16 | bf16)` -- two bf16 values packed into a uint32. This is passed as a runtime argument.
- **Rationale**: This is the required format for `generate_reduce_scaler` and `generate_bcast_scalar_bfloat16` helpers. The MEMORY.md file explicitly warns: "generate_reduce_scaler() and generate_bcast_scalar_bfloat16() expect (bf16 << 16 | bf16) format, NOT IEEE 754 float32."
- **Alternatives Considered**: None -- this is the only correct format.
- **Tradeoffs**: The host must compute the bf16 conversion and packing before passing to the kernel.

## Work Distribution

### Work Unit Definition
One **tile-row** = one horizontal row of tiles spanning the full width of the tensor. Each tile-row contains `Wt` tiles, where `Wt = padded_width / 32`. The tile-row contains 32 rows of elements.

### Parallelization Strategy
- **Grid**: 1x1 (single core)
- **Work per core**: All `Ht_total` tile-rows, where `Ht_total = physical_volume / (padded_width * 32)`. For a tensor of shape `[..., H, W]`, `Ht_total = (product of all dims except last) / 32`.
- **Load balancing**: N/A (single core)

## Data Flow

### High-Level Flow
```
DRAM[RM input] --> Reader --> CB c_0 (RM sticks)
                                   |
                   Compute: tilize --> CB c_1 (tiled)
                                           |
                   Compute: reduce(SUM, ROW, 1/W) --> CB c_2 (mean tile)
                                           |
                   Compute: sub(COL) [c_1 - c_2] --> CB c_3 (centered tiles, PERSISTENT)
                                                         |
                   Compute: square [c_3] --> CB c_24 (squared tiles)
                                                |
                   Compute: reduce(SUM, ROW, 1/W) --> CB c_4 (variance tile)
                                                         |
                   Compute: add(SCALAR) [c_4 + c_7(eps)] --> CB c_25 (var+eps tile)
                                                                  |
                   Compute: rsqrt --> CB c_5 (inv_std tile)
                                         |
                   Compute: mul(COL) [c_3 * c_5] --> CB c_6 (standardized tiles)
                                                          |
                   Compute: untilize --> CB c_16 (RM sticks)
                                              |
                   Writer --> DRAM[RM output]

Reader also generates at startup:
  CB c_8 (reduce scaler = 1/W tile) -- Program lifetime
  CB c_7 (epsilon scalar tile)      -- Program lifetime
```

### Kernel Data Movement

| Kernel | Core | NOC | Actual Function |
|--------|------|-----|-----------------|
| Reader | RISCV_0 | NOC0 | Read RM sticks from DRAM into CB c_0 (32 sticks per tile-row, each stick = W * 2 bytes). Generate scaler tile in CB c_8 and epsilon tile in CB c_7 at startup. |
| Compute | RISCV_2 (TRISC) | N/A | Per tile-row: tilize(c_0->c_1), reduce_row(c_1->c_2), sub_col(c_1,c_2->c_3), square(c_3->c_24), reduce_row(c_24->c_4), add_scalar(c_4,c_7->c_25), rsqrt(c_25->c_5), mul_col(c_3,c_5->c_6), untilize(c_6->c_16) |
| Writer | RISCV_1 | NOC1 | Read RM sticks from CB c_16, write to DRAM output tensor. 32 sticks per tile-row, each stick = W * 2 bytes. |

### Circular Buffer Requirements

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_rm_in | Input RM sticks from reader | Wt tiles | Wt tiles | Single | Reader | Compute (tilize) | Block (per tile-row) |
| c_1 | cb_tilized | Tilized input tiles | Wt tiles | Wt tiles | Single | Compute (tilize) | Compute (reduce, sub) | Block (per tile-row) |
| c_2 | cb_mean | Row mean (reduce output) | 1 tile | 1 tile | Single | Compute (reduce) | Compute (sub) | Block (per tile-row) |
| c_3 | cb_centered | Centered tiles (x - mean) | Wt tiles | Wt tiles | Single | Compute (sub) | Compute (square, mul) | Block (per tile-row, persistent across square+mul) |
| c_24 | cb_squared | Squared centered tiles | Wt tiles | Wt tiles | Single | Compute (square) | Compute (reduce) | Block (per tile-row) |
| c_4 | cb_var | Row variance (reduce output) | 1 tile | 1 tile | Single | Compute (reduce) | Compute (add) | Block (per tile-row) |
| c_25 | cb_var_plus_eps | var + epsilon | 1 tile | 1 tile | Single | Compute (add) | Compute (rsqrt) | Block (per tile-row) |
| c_5 | cb_inv_std | 1/sqrt(var+eps) | 1 tile | 1 tile | Single | Compute (rsqrt) | Compute (mul) | Block (per tile-row) |
| c_6 | cb_result | Standardized tiles (TILE fmt) | Wt tiles | Wt tiles | Single | Compute (mul) | Compute (untilize) | Block (per tile-row) |
| c_16 | cb_rm_out | Output RM sticks for writer | Wt tiles | Wt tiles | Single | Compute (untilize) | Writer | Block (per tile-row) |
| c_7 | cb_eps | Epsilon scalar tile | 1 tile | 1 tile | Single | Reader (once) | Compute (all rows) | Program |
| c_8 | cb_scaler | Reduce scaler tile (1/W) | 1 tile | 1 tile | Single | Reader (once) | Compute (all rows) | Program |

**CB Sizing Notes**:
- All "Wt tiles" CBs: page_size = `tile_size(DataFormat::Float16_b)` = 2048 bytes (for BF16 32x32 tiles). Total = `Wt * 2048` bytes.
- Single tile CBs: 2048 bytes each.
- Peak L1 usage: `(5 * Wt + 7) * 2048` bytes. For Wt=4 (width=128): 27 * 2048 = 55,296 bytes. For Wt=32 (width=1024): 167 * 2048 = 342,016 bytes. Well within L1 capacity (1.5MB).
- All CBs are single-buffered because there is one compute kernel handling everything sequentially. Double-buffering would only help if reader/compute/writer overlap, but the compute kernel is the bottleneck and processes one tile-row fully before moving to the next.

## Memory Access Patterns

### RISCV_0 ("reader" / BRISC) Access

**Startup (once)**:
1. Reserve CB c_8, fill with reduce scaler (1/W as packed bf16) using `generate_reduce_scaler`, push. This tile persists for the entire program.
2. Reserve CB c_7, fill with epsilon value (as packed bf16) using `generate_bcast_scalar_bfloat16`, push. This tile persists for the entire program.

**Per tile-row (Ht_total iterations)**:
3. Pre-compute 32 NOC addresses for the 32 sticks in this tile-row using `get_noc_addr(stick_id, accessor)`.
4. Reserve CB c_0 for Wt tiles worth of space.
5. Read 32 sticks (each of width `W * 2` bytes) from DRAM via `noc_async_read`.
6. Barrier, push Wt tiles to CB c_0.

Pattern: Same as tilize reader. Sequential stick reads within each tile-row group.

### RISCV_1 ("writer" / NCRISC) Access

**Per tile-row (Ht_total iterations)**:
1. Wait for Wt tiles in CB c_16 (one tile-row of RM output).
2. For each of the 32 rows within the tile-row:
   - Compute L1 read address within CB c_16.
   - Compute output page ID (stick ID) from global row number.
   - Write `W * 2` bytes to DRAM via `noc_async_write` with TensorAccessor.
3. Barrier after all 32 rows written.
4. Pop Wt tiles from CB c_16.

Pattern: Same as untilize writer. Sequential row-stick writes per tile-row.

### Compute Access

Per tile-row, the compute kernel performs these phases in sequence:

**Phase 1: Tilize (c_0 -> c_1)**
- `cb_wait_front(c_0, Wt)` -> `tilize_block(c_0, Wt, c_1)` -> `cb_pop_front(c_0, Wt)` -> `cb_push_back(c_1, Wt)`
- Uses `compute_kernel_lib::tilize<c_0, c_1>` helper.

**Phase 2: Reduce mean (c_1 -> c_2)**
- `cb_wait_front(c_1, Wt)` -> reduce SUM with scaler 1/W -> `cb_push_back(c_2, 1)`
- Input tiles NOT popped (using `WaitUpfrontNoPop` policy) -- needed for subtraction.
- Uses `compute_kernel_lib::reduce<SUM, REDUCE_ROW, WaitUpfrontNoPop>`.

**Phase 3: Subtract mean (c_1, c_2 -> c_3)**
- `cb_wait_front(c_2, 1)` -> for each tile: `sub_tiles(c_1[i], c_2[0])` using COL broadcast -> `cb_push_back(c_3, Wt)`
- Input A (c_1) already waited from phase 2, using `NoWaitNoPop` then explicit pop after sub.
- Input B (c_2, mean tile) used with COL broadcast and not popped yet (needed? no, can pop after sub).
- Uses `compute_kernel_lib::sub<COL, NoWaitPopAtEnd, WaitAndPopPerTile>` on c_1 and c_2 to produce c_3.
- After sub completes, c_1 tiles are popped, c_2 (mean) is popped.

**Phase 4: Square centered (c_3 -> c_24)**
- `cb_wait_front(c_3, Wt)` -> square each tile -> `cb_push_back(c_24, Wt)`
- Input tiles NOT popped (using `WaitUpfrontNoPop` policy) -- c_3 needed later for final multiply.
- Uses `compute_kernel_lib::square<WaitUpfrontNoPop>`.

**Phase 5: Reduce variance (c_24 -> c_4)**
- reduce SUM with scaler 1/W on squared tiles -> `cb_push_back(c_4, 1)`
- c_24 tiles consumed and popped (using `WaitAndPopPerTile` or `BulkWaitBulkPop`).
- Uses `compute_kernel_lib::reduce<SUM, REDUCE_ROW>`.

**Phase 6: Add epsilon (c_4 + c_7 -> c_25)**
- `cb_wait_front(c_4, 1)` -> add epsilon tile (c_7, SCALAR broadcast) -> `cb_push_back(c_25, 1)` -> pop c_4.
- c_7 (epsilon) is NOT popped (program lifetime).
- Uses `compute_kernel_lib::add<SCALAR, WaitAndPopPerTile, WaitUpfrontNoPop>`.

**Phase 7: Rsqrt (c_25 -> c_5)**
- `cb_wait_front(c_25, 1)` -> acquire DST -> unpack c_25 -> `rsqrt_tile(0)` -> pack to c_5 -> `cb_push_back(c_5, 1)` -> pop c_25.
- Manual rsqrt using `copy_tile` + `rsqrt_tile` + `pack_tile` pattern.

**Phase 8: Multiply by inv_std (c_3, c_5 -> c_6)**
- c_3 already in CB from phase 4 (not popped). c_5 has 1 tile.
- `cb_wait_front(c_5, 1)` -> for each tile: `mul_tiles(c_3[i], c_5[0])` using COL broadcast -> `cb_push_back(c_6, Wt)`
- c_3 consumed and popped. c_5 popped.
- Uses `compute_kernel_lib::mul<COL, NoWaitPopAtEnd, WaitAndPopPerTile>` with c_3 (already waited, NoWait) and c_5.

**Phase 9: Untilize (c_6 -> c_16)**
- `cb_wait_front(c_6, Wt)` -> untilize_block -> `cb_push_back(c_16, Wt)` -> pop c_6.
- Uses `compute_kernel_lib::untilize<Wt, c_6, c_16>` helper.

## Compile-Time Arguments

### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | stick_size | uint32_t | Size of one RM stick in bytes: `W * 2` (bf16 elements) |
| 1 | cb_rm_in | uint32_t | Input CB ID (c_0) |
| 2 | cb_scaler | uint32_t | Reduce scaler CB ID (c_8) |
| 3 | cb_eps | uint32_t | Epsilon CB ID (c_7) |
| 4+ | TensorAccessorArgs(src) | uint32_t[] | Tensor accessor for input buffer |

### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | Wt | uint32_t | Width in tiles (tiles per tile-row) |
| 1 | Ht_total | uint32_t | Total number of tile-rows to process |
| 2 | cb_rm_in | uint32_t | c_0 - RM input staging |
| 3 | cb_tilized | uint32_t | c_1 - Tilized tiles |
| 4 | cb_mean | uint32_t | c_2 - Row mean tile |
| 5 | cb_centered | uint32_t | c_3 - Centered tiles |
| 6 | cb_squared | uint32_t | c_24 - Squared tiles |
| 7 | cb_var | uint32_t | c_4 - Variance tile |
| 8 | cb_var_plus_eps | uint32_t | c_25 - Var + eps tile |
| 9 | cb_inv_std | uint32_t | c_5 - Inv std tile |
| 10 | cb_result | uint32_t | c_6 - Standardized tiles |
| 11 | cb_rm_out | uint32_t | c_16 - RM output staging |
| 12 | cb_eps | uint32_t | c_7 - Epsilon tile |
| 13 | cb_scaler | uint32_t | c_8 - Reduce scaler tile |

### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_rm_out | uint32_t | Output CB ID (c_16) |
| 1 | output_stick_size | uint32_t | Size of one output RM stick in bytes: `W * 2` |
| 2 | tile_height | uint32_t | Tile height (32) |
| 3 | Wt | uint32_t | Tiles per tile-row (for cb_wait_front/cb_pop_front sizing) |
| 4+ | TensorAccessorArgs(dst) | uint32_t[] | Tensor accessor for output buffer |

## Runtime Arguments

### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Input buffer base address in DRAM |
| 1 | num_sticks | uint32_t | Total number of sticks to read: `Ht_total * 32` |
| 2 | Wt | uint32_t | Tiles per tile-row (for cb_reserve sizing) |
| 3 | start_stick_id | uint32_t | First stick ID (0 for single-core) |
| 4 | packed_reduce_scaler | uint32_t | 1/W packed as (bf16 << 16 \| bf16) |
| 5 | packed_eps | uint32_t | Epsilon packed as (bf16 << 16 \| bf16) |

### Compute Kernel

No runtime arguments (all parameters are compile-time for single-core).

### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Output buffer base address in DRAM |
| 1 | num_tile_rows | uint32_t | Total tile-rows to write: `Ht_total` |
| 2 | start_tile_row | uint32_t | First tile-row index (0 for single-core) |

## Edge Cases
| Condition | Expected Behavior |
|-----------|-------------------|
| Single tile-row (Ht_total=1) | Process one tile-row, output shape same as input |
| Single tile width (Wt=1, W=32) | Mean reduces to single tile's row averages. All operations still valid. |
| Large width (Wt=128, W=4096) | CB memory usage: 5*128*2KB = 1.28MB. Fits in L1 (1.5MB). May need to check. |
| Very large width (Wt>145) | May exceed L1. Should validate and reject at API level. |
| Epsilon = 0 | Division by zero possible for zero-variance rows. Not explicitly guarded; user's responsibility. |
| All-zero row | mean=0, centered=0, var=0, var+eps=eps, rsqrt(eps)=1/sqrt(eps), output=0. Correct. |
| Constant row (all same value) | mean=value, centered=0, var=0, output=0. Correct. |
| 3D tensor [B, H, W] | Treated as [B*H/32, W] in tiles. Each tile-row processed independently. Correct. |
| 4D tensor [B, C, H, W] | Treated as [B*C*H/32, W] in tiles. Each tile-row processed independently. Correct. |
| Rank 2 tensor [H, W] | Simplest case. H/32 tile-rows, each W/32 tiles wide. |
| Width not divisible by 32 | Rejected with error at API level. |
| Height not divisible by 32 | Rejected with error at API level. |

## Agent Handoff

This spec will be consumed by implementation agents. Each agent reads specific sections:

| Agent | Reads These Sections |
|-------|---------------------|
| **ttnn-generic-op-builder** | API Specification, Input Tensor Requirements, Output Tensor Specification, Circular Buffer Requirements, Work Distribution, Compile-Time Arguments, Runtime Arguments |
| **ttnn-kernel-dataflow** (kernel-designer) | Kernel Data Movement, Memory Access Patterns, Compile-Time Arguments, Runtime Arguments, Component Sources |
| **ttnn-kernel-compute** (kernel-designer) | Compute Access (all 9 phases), Mathematical Definition, Circular Buffer Requirements, Component Sources |
| **ttnn-kernel-writer** | All sections (implements actual kernels based on design from kernel-designer) |

The agents know HOW to implement; this spec defines WHAT to implement.

## Test Criteria

What behavior should be verified (agents decide how/when to test):

### Validation Behavior
- Input tensor rank < 2 -> error containing "rank >= 2"
- Input tensor layout is TILE_LAYOUT -> error containing "ROW_MAJOR"
- Input tensor dtype is not BFLOAT16 -> error containing "bfloat16"
- Input tensor not on device -> error containing "on device"
- Input tensor memory layout is sharded -> error containing "interleaved"
- Input tensor last dim not divisible by 32 -> error containing "divisible by 32"

### Shape Behavior
- Output shape equals input shape for all valid inputs
- Output layout is ROW_MAJOR
- Output dtype is BFLOAT16
- Output memory layout is INTERLEAVED

### Functional Behavior
- **Single tile (32x32)**: Compare row_centralize output against PyTorch `(x - x.mean(-1, keepdim=True)) / torch.sqrt(x.var(-1, keepdim=True, correction=0) + eps)`
- **Multi-tile width (e.g., 32x128)**: Same comparison
- **Multi-tile height (e.g., 64x64)**: Same comparison, verify each row independently normalized
- **Batch dimension (e.g., 2x32x64)**: Verify batch elements processed correctly
- **Numerical accuracy**: rtol=0.02, atol=0.02 (bf16 precision). Use `correction=0` (population variance, not sample) in PyTorch reference since we divide by W, not W-1.
- **Zero-variance row**: Output should be all zeros (or near-zero given epsilon)
- **Constant row**: Output should be all zeros

## Open Questions

1. **L1 Memory Limit**: For very wide tensors (Wt > ~145), the CB memory may exceed L1 capacity. Should we add a width limit check in the API validation, or should we rely on the runtime to fail gracefully? **Recommendation**: Add an explicit width check that rejects tensors whose `5 * Wt * tile_size + 7 * tile_size > L1_USABLE_SIZE` (approximately 1.2MB usable out of 1.5MB total).

2. **Padding Support**: Currently requires last two dimensions to be multiples of 32. Should padding/unpadding be added for the initial version? **Recommendation**: No, keep it simple. Add padding support in a future version.

3. **FP32 Support**: Currently only supports BFLOAT16. Should FP32 be supported? **Recommendation**: Not for the initial version. FP32 would require different tile sizes, SFPU compute variant selection, and different scaler packing. Add as a future enhancement.

## References
- Reference analyses:
  - `/localdev/mstaletovic/tt-metal/ttnn/ttnn/operations/row_centralize/agent_logs/tilize_analysis.md`
  - `/localdev/mstaletovic/tt-metal/ttnn/ttnn/operations/row_centralize/agent_logs/untilize_analysis.md`
  - `/localdev/mstaletovic/tt-metal/ttnn/ttnn/operations/row_centralize/agent_logs/batch_norm_analysis.md`
- DeepWiki queries:
  - "How does reduce work in tt-metal for row-wise reductions?" -- Confirmed reduce_tile_math with REDUCE_ROW, scaler tile with 1/N for AVG, generate_reduce_scaler pattern.
- Documentation consulted:
  - `METALIUM_GUIDE.md` -- Hardware architecture, tile-based computing, kernel types
  - `tech_reports/tensor_layouts/tensor_layouts.md` -- RM vs tile layout, page definitions
  - `tech_reports/tensor_accessor/tensor_accessor.md` -- TensorAccessor for DRAM address mapping
  - `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp` -- Unified reduce function API
  - `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp` -- generate_reduce_scaler API
  - `ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp` -- sub, mul, add, square with broadcast support
  - `ttnn/cpp/ttnn/kernel_lib/scalar_helpers.hpp` -- generate_bcast_scalar_bfloat16 API
  - `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp` -- compute_kernel_lib::tilize API
  - `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp` -- compute_kernel_lib::untilize API
  - `tt_metal/include/compute_kernel_api/eltwise_unary/rsqrt.h` -- rsqrt_tile API
