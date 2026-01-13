# Execution Log: Untilize Multi-Core Analysis

## Session Info
- **Started**: Session start
- **Operation**: untilize_multi_core
- **Program Factory Path**: `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/data_movement/untilize/device/factories/untilize_multi_core_program_factory.cpp`
- **Analysis Purpose**: Reference for hybrid operations with row-major interleaved output

## Execution Timeline

### Step 1: Initial Setup and Program Factory Reading
**Action**: Read the main program factory file
**Command/Tool**: Read tool
**File**: `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/data_movement/untilize/device/factories/untilize_multi_core_program_factory.cpp`
**Result**: Successfully read 434 lines
**Key Findings**:
- Two kernel paths for reader: sharded (`reader_unary_sharded.cpp`) vs interleaved (`reader_unary_start_id.cpp`)
- Two compute kernel variants: `pack_untilize_variable_num_blocks.cpp` (fast) and `untilize_variable_num_blocks.cpp` (slow)
- Single writer kernel: `writer_unary_stick_layout_split_rows_multi_core.cpp`
- Uses `split_blocks_for_tilize` for work distribution
- CB c_0 for input, CB c_16 for output
- Supports both interleaved and sharded input/output

**Decision**: Proceed to read all identified kernel files in parallel

### Step 2: Read Table Templates Reference
**Action**: Read table templates for output format
**Command/Tool**: Read tool
**File**: `/localdev/mstaletovic/tt-metal/.claude/references/table-templates.md`
**Result**: Successfully read 82 lines
**Key Findings**:
- Standard formats for Tensor Format Table, CB Table, Core Distribution Table
- Kernel Specification Table format
- Compile-time and Runtime Arguments table formats

### Step 3: Read All Kernel Files
**Action**: Read all 5 kernel files simultaneously
**Command/Tool**: Multiple Read tool calls in parallel
**Files**:
1. `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/reader_unary_start_id.cpp`
2. `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/writer_unary_stick_layout_split_rows_multi_core.cpp`
3. `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/pack_untilize_variable_num_blocks.cpp`
4. `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize_variable_num_blocks.cpp`
5. `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp`

**Results**:
- reader_unary_start_id.cpp: 49 lines - Sequential tile reading with TensorAccessor
- writer_unary_stick_layout_split_rows_multi_core.cpp: 127 lines - Row-by-row stick writing pattern (critical for analysis)
- pack_untilize_variable_num_blocks.cpp: 27 lines - Uses unified untilize helper
- untilize_variable_num_blocks.cpp: 22 lines - Also uses unified untilize helper
- reader_unary_sharded.cpp: 15 lines - Simple cb_push_back for pre-loaded data

**Decision**: The writer kernel is the most important for understanding row-major output. Need to read untilize_helpers.hpp for compute details.

### Step 4: Read Supporting Headers
**Action**: Read untilize helper library and work split header
**Command/Tool**: Multiple Read tool calls in parallel
**Files**:
1. `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp`
2. `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/core/work_split/work_split_tilize.hpp`

**Results**:
- untilize_helpers.hpp: 314 lines - Comprehensive unified untilize implementation with automatic path selection
- work_split_tilize.hpp: 485 lines - BlockSplit struct and split_blocks_for_tilize function

**Key Findings**:
- Untilize helper automatically selects between pack_untilize and standard untilize based on width and data format
- DEST_AUTO_LIMIT is used for width constraints
- BlockSplit contains ncores, core_range, cliff_core_range, nblocks_per_core info
- distribute_work function handles complex padding scenarios

### Step 5: DeepWiki Queries for Architecture Concepts
**Action**: Query DeepWiki for TT-Metal specific concepts
**Command/Tool**: mcp__deepwiki__ask_question (3 parallel queries)

**Query 1**: "How does the untilize operation work in TT-Metal?"
**Response Summary**:
- Three variants: Unpack Untilize, Pack Untilize, DST Untilize
- pack_untilize operates in PACK thread (hardware-accelerated)
- Standard untilize operates in UNPACK thread
- DEST constraints: 8 tiles (half-sync 16-bit), 4 tiles (half-sync 32-bit), 16/8 tiles (full-sync)
**How Used**: Documented in compute kernel section and implementation notes

**Query 2**: "What is the TensorAccessor API in TT-Metal?"
**Response Summary**:
- Abstracts logical-to-physical address mapping
- TensorAccessorArgs configures compile vs runtime parameters
- Supports both interleaved and sharded access patterns
- get_noc_addr() calculates NoC address for page access
**How Used**: Documented in reader/writer kernel descriptions

**Query 3**: "How does split_blocks_for_tilize work?"
**Response Summary**:
- Divides nblocks among available cores
- Returns core_range (full cores) and cliff_core_range (remainder)
- Full cores get nblocks_per_core, cliff core gets nblocks_per_core_cliff
**How Used**: Documented in Core Distribution Strategy section

### Step 6: Additional DeepWiki Queries
**Action**: Query for CB and write pattern specifics
**Command/Tool**: mcp__deepwiki__ask_question (2 parallel queries)

**Query 4**: "What is the create_cb function in TT-Metal?"
**Response Summary**:
- Helper wrapper around CreateCircularBuffer
- Typical indices: c_0, c_1 for inputs, c_16 for output
- Supports single or multiple CB configuration
- Double buffering (2 tiles) enables overlap
**How Used**: Documented in CB Configuration section

**Query 5**: "How does noc_async_write work for row-major data to DRAM interleaved?"
**Response Summary**:
- Creates TensorAccessor with page size = stick_size_bytes
- Pattern: cb_wait_front -> get_read_ptr -> loop (get_noc_addr, noc_async_write) -> barrier -> cb_pop_front
- Page ID calculation maps rows to interleaved pages
**How Used**: Critical for "Memory Access Patterns" and "Usage as Output Stage Reference" sections

### Step 7: Read METALIUM_GUIDE.md
**Action**: Read core architecture documentation
**Command/Tool**: Read tool with limit
**File**: `/localdev/mstaletovic/tt-metal/METALIUM_GUIDE.md`
**Result**: Read first 500 lines
**Key Findings**:
- Three-kernel model: reader, compute, writer
- CB synchronization via cb_reserve_back/cb_push_back/cb_wait_front/cb_pop_front
- NoC0 for reads, NoC1 for writes (typical)
- Tile format: 32x32 with 4 faces of 16x16
- Standard CB indices used in examples match observed pattern (c_0, c_1 inputs, c_16 output)

### Step 8: Read cb_utils.hpp
**Action**: Understand create_cb helper function
**Command/Tool**: Read tool
**File**: `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/cb_utils.hpp`
**Result**: 60 lines
**Key Findings**:
- create_cb is a template wrapper around CreateCircularBuffer
- Returns tuple of (CB index, CBHandle)
- Supports globally allocated addresses for sharded CBs
- Simplified interface compared to raw CreateCircularBuffer

### Step 9: Write Analysis Document
**Action**: Compose comprehensive analysis markdown
**Command/Tool**: Write tool
**File**: `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/data_movement/untilize/device/factories/untilize_multi_core_analysis.md`
**Result**: Successfully created analysis document
**Content**: Complete analysis following template with focus on row-major output patterns

### Step 10: Write Execution Log
**Action**: Document analysis process
**Command/Tool**: Write tool
**File**: `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/data_movement/untilize/device/factories/untilize_multi_core_analyzer_execution_log.md`
**Result**: This file

## Files Read

| File | Purpose | Key Findings |
|------|---------|--------------|
| untilize_multi_core_program_factory.cpp | Main program factory | Kernel paths, CB config, work split, runtime args |
| table-templates.md | Output format | Standard table templates for analysis |
| reader_unary_start_id.cpp | Interleaved reader kernel | TensorAccessor usage, sequential tile reads |
| writer_unary_stick_layout_split_rows_multi_core.cpp | Writer kernel | Row-major stick write pattern (critical) |
| pack_untilize_variable_num_blocks.cpp | Fast compute kernel | Uses unified untilize helper |
| untilize_variable_num_blocks.cpp | Slow compute kernel | Also uses unified helper |
| reader_unary_sharded.cpp | Sharded reader kernel | Simple cb_push_back pattern |
| untilize_helpers.hpp | Compute implementation | Automatic path selection, DEST limits |
| work_split_tilize.hpp | Work distribution | BlockSplit struct, cliff core handling |
| METALIUM_GUIDE.md | Architecture reference | Three-kernel model, CB operations |
| cb_utils.hpp | CB helper | create_cb wrapper function |

## DeepWiki Queries

| Query | Response Summary | How Used |
|-------|------------------|----------|
| Untilize operation variants | pack_untilize vs standard, DEST constraints | Compute kernel section, implementation notes |
| TensorAccessor API | Address abstraction, compile/runtime args | Reader/writer kernel descriptions |
| split_blocks_for_tilize | Full/cliff core separation, work division | Core distribution strategy |
| create_cb function | CB helper, typical indices | CB configuration section |
| noc_async_write pattern | Row-major write sequence | Memory access patterns, output stage reference |

## Errors Encountered

| Error | Context | Resolution |
|-------|---------|------------|
| None | N/A | Analysis completed successfully |

## Key Decisions

| Decision Point | Options Considered | Choice Made | Rationale |
|----------------|-------------------|-------------|-----------|
| Focus area | General analysis vs output-focused | Output-focused | User requested reference for row-major output in hybrid operations |
| Writer kernel emphasis | Equal coverage vs detailed | Detailed writer analysis | Writer implements the row-major output pattern critical for the use case |
| DeepWiki usage | Minimal vs comprehensive | Comprehensive (5 queries) | Needed to verify architectural assumptions and document sources |
| Output format | Brief vs detailed tables | Detailed with examples | Better serves as reference for implementation |

## Files Created/Modified

| File | Action | Description |
|------|--------|-------------|
| untilize_multi_core_analysis.md | Created | Comprehensive analysis document |
| untilize_multi_core_analyzer_execution_log.md | Created | This execution log |

## Final Status
- **Completed**: Yes
- **Output File**: `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/data_movement/untilize/device/factories/untilize_multi_core_analysis.md`
- **Issues**: None

## Summary

The analysis successfully documented the untilize multi-core operation with special emphasis on:

1. **Row-Major Output Pattern**: Detailed the writer kernel's stick-by-stick write pattern, which is the key pattern for any operation outputting row-major interleaved data.

2. **CB Configuration**: Documented how CB c_16 is used for row-major output staging with appropriate buffering.

3. **Index Calculations**: Explained how tile-space indices map to row-major page IDs.

4. **TensorAccessor Usage**: Documented how the writer kernel uses TensorAccessor with stick_size for page-based DRAM writes.

5. **Usage Guidance**: Added a "Usage as Output Stage Reference" section with practical guidance for creating hybrid operations.
