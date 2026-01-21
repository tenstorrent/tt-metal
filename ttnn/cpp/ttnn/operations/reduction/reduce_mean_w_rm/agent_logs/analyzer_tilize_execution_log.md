# Execution Log: Tilize Multi-Core Interleaved Analysis

## Session Info
- **Started**: 2026-01-21
- **Operation**: tilize_multi_core_interleaved
- **Program Factory Path**: `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_multi_core_interleaved_program_factory.cpp`

## Execution Timeline

### Step 1: Read Program Factory
**Action**: Read the main program factory file
**Command/Tool**: Read tool
**Result**: Successfully read 197 lines. Key observations:
- Uses `split_blocks_for_tilize` for work distribution
- Creates two CBs: c_0 (input) and c_16 (output)
- Reader kernel: `reader_unary_stick_layout_split_rows_interleaved.cpp`
- Writer kernel: `writer_unary_interleaved_start_id.cpp` (reused from eltwise/unary)
- Compute kernel: `tilize.cpp` (deprecated location)
- Uses TensorAccessorArgs for address generation

**Decision**: Need to read all three kernel files to understand complete data flow

### Step 2: Read Logging Config
**Action**: Verify breadcrumbs logging is enabled
**Command/Tool**: Read tool
**Result**: JSON config confirmed logging is enabled with "detailed" verbosity

### Step 3: Read Table Templates
**Action**: Read table-templates.md for consistent output format
**Command/Tool**: Read tool
**Result**: Retrieved 82 lines of table templates for: Tensor Format, Circular Buffer, Core Distribution, Arguments, Kernel Specification tables

### Step 4: Read Kernel Files (Parallel)
**Action**: Read all three kernel files simultaneously
**Command/Tool**: Read tool (3 parallel calls)

**Reader Kernel Result** (reader_unary_stick_layout_split_rows_interleaved.cpp):
- 52 lines, key insights:
- Uses `TensorAccessor` with compile-time args
- Reads 32 sticks (rows) at a time to form tile height
- Pre-computes base_src_noc_addr for all 32 rows before reading
- `read_tiles` lambda handles CB operations
- Block-based iteration: outer loop over blocks, inner reads full tile rows

**Writer Kernel Result** (writer_unary_interleaved_start_id.cpp):
- 41 lines, key insights:
- Generic writer, works for both TILE and ROW_MAJOR
- Page-at-a-time write pattern
- Has BACKWARDS and OUT_SHARDED preprocessor variants
- Uses `noc_async_write_page` with TensorAccessor

**Compute Kernel Result** (tilize.cpp):
- 20 lines, minimal wrapper using `tilize_helpers.hpp`
- Calls `compute_kernel_lib::tilize()` unified function
- Two compile-time args: `per_core_block_cnt` and `per_core_block_tile_cnt`

**Decision**: Need to read tilize_helpers.hpp to understand compute implementation

### Step 5: Read Work Split Utility
**Action**: Read work_split_tilize.hpp
**Command/Tool**: Read tool
**Result**: 493 lines, comprehensive work distribution utilities:
- `BlockSplit` struct: stores core ranges and blocks per core
- `split_blocks_for_tilize`: computes ncores, nblocks_per_core, cliff handling
- `BlockSplitWH` for 2D sharding scenarios (not used in this factory)
- `BlockRep` for complex padding patterns
- Key formula: `nblocks_per_core = ceil(nblocks / grid_area)`

**Decision**: This operation uses 1D block distribution

### Step 6: Read tilize_helpers.hpp
**Action**: Understand compute kernel implementation details
**Command/Tool**: Read tool
**Result**: 216 lines, unified tilize function:
- Template parameters: init, uninit, use_fast, use_dt, skip_wait
- Main loop: wait for input, reserve output, tilize_block, push/pop
- Supports variable row alignment and asymmetric input/output counts
- Uses `tilize_block` from `compute_kernel_api/tilize.h`

### Step 7: Read cb_utils.hpp
**Action**: Understand CB creation helper
**Command/Tool**: Read tool
**Result**: 60 lines:
- `create_cb` helper simplifies CircularBufferConfig setup
- Parameters: CB index, program, core_spec, page_size, num_pages, data_format
- Optional buffer parameter for globally allocated addresses

### Step 8: Read Program Factory Header
**Action**: Understand shared_variables_t structure
**Command/Tool**: Read tool
**Result**: 29 lines, defines:
- `shared_variables_t`: kernel handles, core list, ncores
- Used for `override_runtime_arguments` to update buffer addresses

### Step 9: DeepWiki Query - Tilize Operation
**Action**: Get high-level understanding of tilize
**Query**: "What is the tilize operation and how does it transform row-major data into tiled format?"
**Result**: Key findings:
- Transforms ROW_MAJOR_LAYOUT to TILE_LAYOUT
- 32x32 tiles stored contiguously
- Tiles split into 16x16 faces internally
- Hardware processes tiles natively for efficiency

### Step 10: DeepWiki Query - TensorAccessor
**Action**: Understand address generation mechanism
**Query**: "What does the TensorAccessor do in tt-metal kernels?"
**Result**: Key findings:
- Abstracts physical memory layout
- Maps logical indices to physical addresses
- `get_noc_addr(page_id)` computes NoC address from page ID
- Handles bank interleaving automatically

### Step 11: DeepWiki Query - tilize_block
**Action**: Understand compute kernel hardware operations
**Query**: "What does tilize_block do in the compute kernel?"
**Result**: Key findings:
- Coordinates unpacker, math core, packer
- `llk_unpack_tilize_block`: reads from input CB
- `llk_math_eltwise_unary_datacopy`: moves data through math core
- `llk_pack`: writes to output CB
- Per-tile iteration within block

### Step 12: DeepWiki Query - Sticks and Rows
**Action**: Understand row-major data terminology
**Query**: "What is the relationship between sticks and rows in row-major layout?"
**Result**: Key findings:
- "Stick" = one row of data (pixel with channel dimension in CNN context)
- Row-major: rows stored contiguously
- Reader fetches sticks, sends to compute via CB
- 32 sticks form one tile height

### Step 13: DeepWiki Query - Circular Buffers with Tilize
**Action**: Understand CB configuration relationship
**Query**: "How do circular buffers work with tilize?"
**Result**: Key findings:
- Page size set to single_tile_size for tilize
- CB capacity = ntiles_per_block * tile_size
- Ensures tile-aligned reads/writes
- tilize_block processes tile-sized chunks

### Step 14: DeepWiki Query - Block Distribution
**Action**: Understand split_blocks_for_tilize
**Query**: "What is a block in the context of tilize operations?"
**Result**: Key findings:
- Block = collection of tiles processed together by one core
- `split_blocks_for_tilize` distributes blocks across cores
- Handles "cliff" core for remainder blocks
- Returns BlockSplit struct with core ranges

### Step 15: Read Tensor Layouts Tech Report
**Action**: Get authoritative tensor layout documentation
**Command/Tool**: Read tool
**Result**: 119 lines, key insights:
- Row-major: each row is a page
- Tiled: 32x32 tiles are pages
- Faces: 16x16 sub-tiles within tile
- Interleaved: round-robin page distribution across banks

### Step 16: Read METALIUM_GUIDE.md (partial)
**Action**: Verify circular buffer and kernel architecture understanding
**Command/Tool**: Read tool (first 200 lines)
**Result**: Confirmed architecture understanding:
- Reader -> CB -> Compute -> CB -> Writer
- Circular buffers for producer-consumer synchronization
- Three kernel types coordinated by CBs
- NoC0 for reads, NoC1 for writes

### Step 17: Read Device Operation Types
**Action**: Understand operation attributes structure
**Command/Tool**: Read tool
**Result**: 41 lines:
- `operation_attributes_t`: output_mem_config, output_dtype, use_multicore, etc.
- `tensor_args_t`: input_tensor, optional_input_tensor
- `shared_variables_t`: kernel handles for runtime arg override

## Files Read

| File | Purpose | Key Findings |
|------|---------|--------------|
| tilize_multi_core_interleaved_program_factory.cpp | Main program factory | Uses split_blocks_for_tilize, creates 2 CBs, 3 kernels |
| reader_unary_stick_layout_split_rows_interleaved.cpp | Reader kernel | Reads 32 sticks at a time, pre-computes NoC addresses |
| writer_unary_interleaved_start_id.cpp | Writer kernel | Generic tile/page writer, page-at-a-time |
| tilize.cpp | Compute kernel | Wrapper calling tilize_helpers.hpp |
| work_split_tilize.hpp | Work distribution | BlockSplit struct, cliff handling, 1D distribution |
| tilize_helpers.hpp | Compute implementation | Unified tilize function with templates |
| cb_utils.hpp | CB creation helper | Simplifies CircularBufferConfig |
| tilize_multi_core_interleaved_program_factory.hpp | Header | shared_variables_t definition |
| tilize_device_operation_types.hpp | Operation types | Attributes and tensor args |
| tensor_layouts/tensor_layouts.md | Tech report | Row-major vs tiled, faces, interleaved |
| METALIUM_GUIDE.md | Architecture guide | Kernel coordination, CB patterns |
| table-templates.md | Output formatting | Standard table formats |

## DeepWiki Queries

| Query | Response Summary | How Used |
|-------|------------------|----------|
| Tilize operation overview | Transforms row-major to tiled format, 32x32 tiles | Foundation for analysis |
| TensorAccessor functionality | Abstracts memory layout, computes NoC addresses | Index calculation section |
| tilize_block details | Unpacker->math->packer coordination | Kernel implementation section |
| Sticks vs rows | Stick = row, 32 sticks per tile height | Data flow understanding |
| CB configuration for tilize | Page size = tile size, capacity in tiles | CB table population |
| Block distribution | split_blocks_for_tilize work allocation | Core distribution section |

## Key Decisions

| Decision Point | Options Considered | Choice Made | Rationale |
|----------------|-------------------|-------------|-----------|
| Work unit granularity | tile vs block vs row | Block (tile rows) | Reader processes 32 rows at a time to form tile height |
| CB buffering classification | Single vs Double | Single-buffered | Capacity equals block size (ntiles_per_block) |
| Core distribution type | 1D vs 2D | 1D | Uses split_blocks_for_tilize (not _wh variant) |
| Reader pattern | Split reader vs standard | Standard | Only RISCV_0 reads, no split pattern |

## Files Created/Modified

| File | Action | Description |
|------|--------|-------------|
| analyzer_tilize_execution_log.md | Created | This execution log |
| tilize_analysis.md | Created | Main analysis output |

## Final Status
- **Completed**: Yes
- **Output File**: `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/reduction/reduce_mean_w_rm/references/tilize_analysis.md`
- **Issues**: None - analysis completed successfully
