# Execution Log: Untilize Multi-Core Analysis

## Session Info
- **Started**: Session start
- **Operation**: untilize_multi_core
- **Program Factory Path**: `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/data_movement/untilize/device/factories/untilize_multi_core_program_factory.cpp`

## Execution Timeline

### Step 1: Initial File Discovery and Reading
**Action**: Read program factory, logging config, and table templates
**Command/Tool**: Read tool (3 parallel reads)
**Result**:
- Logging config confirmed breadcrumbs enabled with detailed verbosity
- Program factory is 424 lines, handles both interleaved and sharded input
- Table templates provide standard format for analysis output
**Decision**: Proceed with kernel discovery

### Step 2: Identify Associated Kernel Files
**Action**: Glob for kernel files in the untilize operation directory
**Command/Tool**: Glob for `*.cpp` and `*.hpp` in kernels directory
**Result**:
Found 12 kernel files:
- `reader_unary_start_id.cpp` (interleaved reader)
- `writer_unary_stick_layout_split_rows_multi_core.cpp` (multi-core writer)
- `untilize_variable_num_blocks.cpp` (standard compute)
- `pack_untilize_variable_num_blocks.cpp` (fast compute)
- Plus several other variants (single core, WH, parallel columns)
**Decision**: Focus on the three kernels used by multi-core factory

### Step 3: Read Reader Kernel (Sharded)
**Action**: Read sharded reader kernel
**Command/Tool**: Read `reader_unary_sharded.cpp`
**Result**:
```cpp
void kernel_main() {
    uint32_t num_tiles_per_core = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    cb_push_back(cb_id_in0, num_tiles_per_core);
}
```
**Decision**: This is a minimal kernel - data already in L1, just makes it visible to CB. This is an optimization pattern for sharded inputs.

### Step 4: Read Reader and Writer Kernels
**Action**: Read interleaved reader, multi-core writer, and both compute kernels
**Command/Tool**: Read tool (4 parallel reads)
**Result**:
- **Reader (interleaved)**: Uses TensorAccessor, reads tiles sequentially with barrier after each
- **Writer**: Complex logic handling input/output block width mismatches, byte-level offsets
- **Both compute kernels**: Both use `compute_kernel_lib::untilize<>()` from unified helper

**Decision**: Need to understand the untilize_helpers.hpp for compute logic

### Step 5: Read Untilize Helpers
**Action**: Read kernel helper library
**Command/Tool**: Read `untilize_helpers.hpp`
**Result**:
314 lines of unified untilize logic with automatic dispatch:
- Pack untilize for narrow widths (within DEST limit)
- Block-based pack untilize for wide integer types
- Standard untilize fallback for wide float types

Key finding: DEST limit is auto-detected from JIT headers
**Decision**: This is sophisticated - need to query DeepWiki about concepts

### Step 6: DeepWiki Query - Untilize Operation
**Action**: Query DeepWiki about untilize operation basics
**Command/Tool**: mcp__deepwiki__ask_question
**Query**: "What is the untilize operation in tt-metal? What does it do and what are the different methods of untilizing (standard untilize vs pack_untilize)?"
**Response Summary**:
- Converts TILE_LAYOUT to ROW_MAJOR_LAYOUT
- Pack untilize is hardware-accelerated, preferred
- Standard untilize is fallback for UINT16, wide FLOAT32
**How Used**: Formed basis of Overview section and kernel selection logic explanation

### Step 7: DeepWiki Queries - TensorAccessor and Layouts
**Action**: Query about TensorAccessor and layout concepts
**Command/Tool**: mcp__deepwiki__ask_question (2 parallel queries)
**Queries**:
1. "What is TensorAccessor in tt-metal and how does it work?"
2. "What is TILE_LAYOUT vs ROW_MAJOR_LAYOUT in tt-metal?"

**Response Summary**:
1. TensorAccessor abstracts physical memory layout, uses DistributionSpec, provides get_noc_addr()
2. Tiles are 32x32 with 16x16 faces, row-major stores consecutive row elements

**How Used**: Documented in Index Calculations and Tensor Format sections

### Step 8: DeepWiki Query - Work Splitting
**Action**: Query about split_work_to_cores and cliff cores
**Command/Tool**: mcp__deepwiki__ask_question
**Query**: "What is the split_work_to_cores function in tt-metal? How does it distribute work across cores and what is the cliff core concept?"
**Response Summary**:
- Distributes work units across available cores
- Cliff cores handle remainder when work doesn't divide evenly
- split_blocks_for_tilize is specialized version for tilize/untilize
**How Used**: Core Distribution Strategy section

### Step 9: Read Supporting Infrastructure
**Action**: Read work_split_tilize.hpp and cb_utils.hpp
**Command/Tool**: Read tool (2 reads)
**Result**:
- **work_split_tilize.hpp**: 493 lines, BlockSplit struct with ncores, core_range, cliff_core_range, nblocks_per_core
- **cb_utils.hpp**: 60 lines, create_cb helper wrapping CircularBufferConfig

**Decision**: Have enough context for work distribution and CB creation

### Step 10: Read Tensor Layouts Documentation
**Action**: Read tensor layouts tech report
**Command/Tool**: Read `tech_reports/tensor_layouts/tensor_layouts.md`
**Result**:
- Row-major: each row is a page
- Tiled: 32x32 tiles split into 16x16 faces
- Face storage: row-major within tile (face0->face1->face2->face3)
**How Used**: Layout Transformation section explanation

### Step 11: Read TensorAccessor Documentation
**Action**: Read tensor accessor tech report
**Command/Tool**: Read `tech_reports/tensor_accessor/tensor_accessor.md`
**Result**:
- Comprehensive guide to TensorAccessor usage
- Host creates TensorAccessorArgs from Buffer
- Device creates TensorAccessor from args + address + page_size
- get_noc_addr() maps logical to physical
**How Used**: Index Calculations and Documentation References sections

### Step 12: DeepWiki Query - CB Creation
**Action**: Query about create_cb and CB indices
**Command/Tool**: mcp__deepwiki__ask_question
**Query**: "What is the create_cb function in tt-metal and what do CB indices like c_0 and c_16 mean?"
**Response Summary**:
- CB indices 0-31 identify buffers per core
- c_0 commonly input, c_16 commonly output (convention)
- CircularBufferConfig specifies size, data format, page size
**How Used**: Circular Buffer Configuration section

### Step 13: Read dest_helpers.hpp
**Action**: Read DEST register capacity helper
**Command/Tool**: Read `dest_helpers.hpp`
**Result**:
105 lines defining DEST capacity detection:
- SyncFull + 16-bit: 16 tiles
- SyncFull + 32-bit: 8 tiles
- SyncHalf + 16-bit: 8 tiles
- SyncHalf + 32-bit: 4 tiles
**How Used**: Explained DEST limit in compute kernel section

### Step 14: Read Header and Types Files
**Action**: Read program factory header and operation types
**Command/Tool**: Read tool (2 reads)
**Result**:
- **Header**: Defines UntilizeMultiCoreProgramFactory with create() and override_runtime_arguments()
- **Types**: untilize_shared_variables_t stores reader_kernel_id, writer_kernel_id, cb_src0, cb_output, cores_with_runtime_args

**Decision**: Have complete picture of operation structure

### Step 15: DeepWiki Query - compute_kernel_hw_startup
**Action**: Query about hardware initialization
**Command/Tool**: mcp__deepwiki__ask_question
**Query**: "What is the compute_kernel_hw_startup function in tt-metal compute kernels?"
**Response Summary**:
- Initializes UNPACK, MATH, PACK hardware units
- Must be called once at kernel start
- Takes CB IDs matching subsequent operations
**How Used**: Kernel Implementations section

### Step 16: Create Output Directories
**Action**: Create directories for analysis output
**Command/Tool**: Bash mkdir -p
**Result**: Created `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/reduction/reduce_mean_w_rm/references/`
**Decision**: Proceed to write analysis document

### Step 17: Write Analysis Document
**Action**: Write comprehensive analysis markdown
**Command/Tool**: Write tool
**Result**: Created `untilize_analysis.md` with all sections:
- Overview, Work Unit, Tensor Format, Data Flow, CB Configuration
- Pipeline Patterns, Index Calculations, Memory Access
- Core Distribution, Arguments, Kernel Implementations
- Implementation Notes, External Sources

## Files Read

| File | Purpose | Key Findings |
|------|---------|--------------|
| `untilize_multi_core_program_factory.cpp` | Main analysis target | 424 lines, handles interleaved/sharded, uses split_blocks_for_tilize |
| `reader_unary_sharded.cpp` | Sharded reader kernel | Minimal - just cb_push_back, data already in L1 |
| `reader_unary_start_id.cpp` | Interleaved reader kernel | Uses TensorAccessor, sequential tile reads |
| `writer_unary_stick_layout_split_rows_multi_core.cpp` | Writer kernel | Complex partial write logic for mismatched sharding |
| `untilize_variable_num_blocks.cpp` | Standard compute kernel | Uses unified untilize helper |
| `pack_untilize_variable_num_blocks.cpp` | Fast compute kernel | Uses unified untilize helper |
| `untilize_helpers.hpp` | Compute helper library | 314 lines, auto-dispatch based on width/type |
| `dest_helpers.hpp` | DEST capacity detection | 4-16 tiles based on sync/accum mode |
| `work_split_tilize.hpp` | Work distribution | BlockSplit struct, cliff core pattern |
| `cb_utils.hpp` | CB creation helper | Wraps CircularBufferConfig |
| `tensor_layouts.md` | Layout documentation | Tile faces, row-major vs tiled |
| `tensor_accessor.md` | TensorAccessor guide | Host/device usage patterns |
| `untilize_multi_core_program_factory.hpp` | Factory header | Defines shared_variables_t |
| `untilize_device_operation_types.hpp` | Operation types | operation_attributes_t, tensor_args_t |
| `logging_config.json` | Breadcrumbs config | Confirmed detailed logging enabled |
| `table-templates.md` | Output format | Standard table formats |

## DeepWiki Queries

| Query | Response Summary | How Used |
|-------|------------------|----------|
| Untilize operation basics | TILE to ROW_MAJOR, pack vs standard | Overview section |
| TensorAccessor functionality | Abstracts memory layout, get_noc_addr() | Index Calculations |
| TILE_LAYOUT vs ROW_MAJOR | 32x32 tiles with faces vs consecutive rows | Layout Transformation |
| split_work_to_cores | Work distribution, cliff cores | Core Distribution |
| create_cb and CB indices | Indices 0-31, CircularBufferConfig | CB Configuration |
| compute_kernel_hw_startup | HW init for UNPACK/MATH/PACK | Kernel Implementations |

## Errors Encountered

| Error | Context | Resolution |
|-------|---------|------------|
| None | - | - |

## Key Decisions

| Decision Point | Options Considered | Choice Made | Rationale |
|----------------|-------------------|-------------|-----------|
| Which kernels to analyze | All 12 kernel files vs 3 used by multi-core | 3 used by multi-core | Focus on specific factory being analyzed |
| How to explain compute dispatch | Detailed algorithm vs high-level | Both - algorithm in notes, high-level in main | Different readers need different depth |
| CB sizing explanation | Just values vs conditional logic | Include conditional logic | Important for understanding when double-buffering is used |
| Sharded vs interleaved coverage | Focus on one vs both | Both with clear separation | Factory handles both, analysis should too |

## Files Created/Modified

| File | Action | Description |
|------|--------|-------------|
| `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/reduction/reduce_mean_w_rm/references/untilize_analysis.md` | Created | Comprehensive analysis document |
| `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/reduction/reduce_mean_w_rm/agent_logs/analyzer_untilize_execution_log.md` | Created | This execution log |

## Final Status
- **Completed**: Yes
- **Output File**: `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/reduction/reduce_mean_w_rm/references/untilize_analysis.md`
- **Issues**: None

## Analysis Insights

### Key Patterns Discovered

1. **Sharded Input Optimization**: When input is sharded, the reader kernel is trivial (just cb_push_back). This is a significant optimization - no actual data movement, just making existing L1 data visible to CB.

2. **Unified Compute Helper**: Both compute kernels delegate to the same `compute_kernel_lib::untilize<>()` function. The difference is which one gets selected by the factory based on data type and width.

3. **Complex Writer for Flexible Sharding**: The writer handles input/output sharding mismatches with nested loops and partial writes. This flexibility comes at the cost of code complexity.

4. **Cliff Core Separation**: A separate kernel instance is created for the cliff core to avoid runtime conditionals in the main processing loop.

### Design Quality Observations

**Strengths**:
- Clean separation of interleaved vs sharded paths
- Unified helper library reduces duplication
- TensorAccessor abstracts memory complexity

**Pain Points**:
- CB sizing logic is complex with multiple conditionals
- Writer kernel has nested loops that are hard to follow
- FLOAT32 limitations require fallback path

### Recommendations for New Operations

1. Use `split_blocks_for_tilize()` for tilize/untilize work distribution
2. Consider sharded input optimization (trivial reader)
3. Use unified helper libraries when available
4. Document CB sizing rationale clearly
5. Test both single-block (no double buffering) and multi-block (double buffering) cases
