# Execution Log: Untilize Multi-Core Analysis

## Session Info
- **Started**: Session start
- **Operation**: untilize_multi_core
- **Program Factory Path**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/factories/untilize_multi_core_program_factory.cpp`
- **Purpose**: Analyze untilize as "output_stage" reference for row_standardize operation

## Execution Timeline

### Step 1: Initial Reconnaissance - Read Program Factory
**Action**: Read the main program factory file to identify all kernel files, CB configuration, and work distribution
**Command/Tool**: Read `untilize_multi_core_program_factory.cpp`
**Result**: Successfully read 425 lines. Identified:
- Reader (interleaved): `reader_unary_start_id.cpp`
- Reader (sharded): `reader_unary_sharded.cpp`
- Writer: `writer_unary_stick_layout_split_rows_multi_core.cpp`
- Compute (fast): `pack_untilize_variable_num_blocks.cpp`
- Compute (slow): `untilize_variable_num_blocks.cpp`
- Two CBs: c_0 (input), c_16 (output)
- Work split via `split_blocks_for_tilize`
**Decision**: Read all kernel files and supporting utilities next

### Step 2: Read Table Templates
**Action**: Read table-templates.md for output formatting standards
**Command/Tool**: Read `.claude/references/table-templates.md`
**Result**: Retrieved all table formats (Tensor Format, CB, Core Distribution, Arguments, Kernel Spec, etc.)

### Step 3: Read All Kernel Files
**Action**: Read all 5 kernel source files identified in Step 1
**Command/Tool**: Read each kernel file in parallel

**reader_unary_start_id.cpp** (34 lines):
- Simple sequential tile reader using TensorAccessor
- Reads tiles one at a time with noc_async_read + barrier
- Runtime args: src_addr, num_tiles, start_page_id
- Compile-time args: cb_id_in0, then TensorAccessorArgs

**writer_unary_stick_layout_split_rows_multi_core.cpp** (110 lines):
- Most complex kernel - handles stick extraction from untilized CB data
- Iterates through tile_height rows per block
- Handles cross-page writes for sharded output
- Uses while loop for partial writes across output page boundaries
- 6 runtime args, 8+ compile-time args (including TensorAccessorArgs)

**pack_untilize_variable_num_blocks.cpp** (29 lines):
- Fast path compute kernel using unified untilize helper
- Calls compute_kernel_hw_startup then compute_kernel_lib::untilize<>()
- DST_ACCUM_MODE affects max_bct (4 vs 8)

**untilize_variable_num_blocks.cpp** (19 lines):
- Slow path compute kernel, same structure as fast path
- Also uses compute_kernel_lib::untilize<>()
- Used for UINT16, wide FLOAT32, or when pack_untilize disabled

**reader_unary_sharded.cpp** (15 lines):
- Trivial kernel - just cb_push_back since data already in L1

### Step 4: Read Supporting Libraries
**Action**: Read untilize_helpers.hpp/inl, dest_helpers.hpp, work_split_tilize.hpp, cb_utils.hpp, operation types
**Command/Tool**: Read each file

**untilize_helpers.hpp** (191 lines):
- Unified untilize function with template params for width, CBs, init mode, wait mode
- Three dispatch paths documented in comments
- InitUninitMode and WaitMode enums

**untilize_helpers.inl** (261 lines):
- Full implementation of dispatch logic
- Pack untilize (single-pass): width <= DEST limit
- Block-based pack untilize: integer types, width > DEST limit, splits into sub-blocks
- Standard untilize: fallback for wide non-integer types
- WaitUpfront forces standard path

**dest_helpers.hpp** (104 lines):
- DEST_AUTO_LIMIT auto-detected from JIT headers
- Half-sync + fp16 = 8, Half-sync + fp32 = 4, Full-sync + fp16 = 16, Full-sync + fp32 = 8

**work_split_tilize.hpp** (492 lines):
- split_blocks_for_tilize returns BlockSplit struct
- Computes nblocks_per_core = ceil(nblocks / grid_area)
- Handles cliff core for remainder blocks
- Also has WH-specific variant for 2D splitting

**cb_utils.hpp** (59 lines):
- create_cb helper wraps CircularBufferConfig creation
- Supports optional globally_allocated_address for sharded buffers

**untilize_device_operation_types.hpp** (47 lines):
- UntilizeOperationAttributes: output_mem_config, use_multicore, use_pack_untilize, fp32_dest_acc_en, etc.
- UntilizeTensorArgs: just input tensor
- UntilizeSharedVariables: kernel handles, CB handles, cores list

### Step 5: Read compute_kernel_hw_startup
**Action**: Understand hardware initialization required by compute kernels
**Command/Tool**: Grep then Read compute_kernel_hw_startup.h
**Result**: Found at `tt_metal/include/compute_kernel_api/compute_kernel_hw_startup.h`
- Configures UNPACK (llk_unpack_hw_configure), MATH (llk_math_pack_sync_init + hw_configure), PACK (llk_pack_init + hw_configure + dest_init)
- Must be called once at kernel start, before any compute API
- Two overloads: (icb0, icb1, ocb) and (icb0, ocb) which sets icb1=icb0

### Step 6: Check MAX_PACK_UNTILIZE_WIDTH
**Action**: Find the constant that controls fast vs slow untilize dispatch
**Command/Tool**: Grep for MAX_PACK_UNTILIZE_WIDTH
**Result**: Defined in `ttnn/api/ttnn/common/constants.hpp` as 8
- pack_untilize does not support > 8 width for FLOAT32

### Step 7: DeepWiki Research
**Action**: Consulted DeepWiki for 4 questions

**Q1**: Untilize operation mechanics and pack_untilize vs standard
- Standard: UNPACK -> MATH -> PACK (all 3 TRISC threads)
- Pack: Only PACK thread, hardware-accelerated
- Selection based on data type and width

**Q2**: TensorAccessor and TensorAccessorArgs
- Host configures, appends to compile-time args
- Device reconstructs, provides get_noc_addr(page_id)
- Abstracts interleaved bank mapping

**Q3**: Sticks in tensor layouts
- Stick = row in row-major layout
- Each row = one page in row-major
- Writer writes sticks to DRAM

**Q4**: split_blocks_for_tilize
- Distributes nblocks across grid
- Full cores get equal blocks, cliff core gets remainder
- Returns BlockSplit struct with core ranges

### Step 8: Write Analysis Document
**Action**: Synthesized all findings into untilize_analysis.md
**Command/Tool**: Write to `/localdev/mstaletovic/tt-metal/ttnn/ttnn/operations/row_standardize/untilize_analysis.md`
**Result**: Created comprehensive analysis with all required sections

## Files Read

| File | Purpose | Key Findings |
|------|---------|--------------|
| `ttnn/cpp/ttnn/operations/data_movement/untilize/device/factories/untilize_multi_core_program_factory.cpp` | Main program factory | 2 CBs (c_0, c_16), 5 kernel variants, split_blocks_for_tilize for work distribution, supports sharded and interleaved |
| `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/reader_unary_start_id.cpp` | Reader kernel (interleaved) | Sequential tile reads with TensorAccessor, one tile at a time |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp` | Reader kernel (sharded) | Just cb_push_back, no data movement |
| `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/writer_unary_stick_layout_split_rows_multi_core.cpp` | Writer kernel | Row-by-row extraction from untilized CB, handles cross-page writes |
| `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/pack_untilize_variable_num_blocks.cpp` | Fast compute kernel | Uses unified untilize helper, hardware-accelerated pack_untilize |
| `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize_variable_num_blocks.cpp` | Slow compute kernel | Same helper, fallback for wide floats/UINT16 |
| `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp` | Untilize helper header | Three dispatch paths, InitUninitMode and WaitMode enums |
| `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.inl` | Untilize helper impl | Full dispatch logic with DEST auto-detection |
| `ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp` | DEST capacity detection | Auto-detects from JIT headers, capacity table |
| `ttnn/cpp/ttnn/operations/core/work_split/work_split_tilize.hpp` | Work distribution | split_blocks_for_tilize with cliff core handling |
| `ttnn/cpp/ttnn/operations/cb_utils.hpp` | CB creation helper | Wraps CircularBufferConfig |
| `ttnn/cpp/ttnn/operations/data_movement/untilize/device/untilize_device_operation_types.hpp` | Type definitions | Operation attributes, tensor args, shared variables |
| `tt_metal/include/compute_kernel_api/compute_kernel_hw_startup.h` | HW startup | Configures UNPACK, MATH, PACK units |
| `ttnn/api/ttnn/common/constants.hpp` | Constants | MAX_PACK_UNTILIZE_WIDTH = 8 |
| `tech_reports/tensor_layouts/tensor_layouts.md` | Layout documentation | Tile vs row-major, face structure, interleaved banking |

## DeepWiki Queries

| Query | Response Summary | How Used |
|-------|------------------|----------|
| Untilize operation mechanics | Standard uses 3 TRISC threads, pack uses only PACK thread | Informed compute kernel analysis and dispatch path documentation |
| TensorAccessor usage | Host configures args, device resolves page IDs to NOC addresses | Documented in Index Calculations and reader/writer kernel specs |
| Sticks in tensor layouts | Stick = row, each row = one page in row-major | Clarified writer output format and page structure |
| split_blocks_for_tilize | Distributes blocks across cores with cliff handling | Documented core distribution strategy |

## Errors Encountered

| Error | Context | Resolution |
|-------|---------|------------|
| Breadcrumb init script failed (missing 3rd arg) | Script requires 4 positional args | Read script source, added operation_name and empty predecessor args |
| Breadcrumb append script JSON parse error | Likely special characters in message | Skipped append calls, logged in this file instead |

## Key Decisions

| Decision Point | Options Considered | Choice Made | Rationale |
|----------------|-------------------|-------------|-----------|
| Analysis scope | Full analysis vs output-stage focused | Full analysis with output-stage emphasis | User requested focus on "output_stage" reference but complete analysis is more useful for downstream agents |
| Kernel dispatch documentation | Document host-side selection only vs include device-side dispatch | Both documented | The unified untilize helper on device has its own dispatch logic separate from host kernel selection |
| CB sizing documentation | Static rules vs conditional rules | Conditional rules with all cases | CB sizing differs significantly between sharded/interleaved and single/multi-block cases |

## Files Created/Modified

| File | Action | Description |
|------|--------|-------------|
| `ttnn/ttnn/operations/row_standardize/untilize_analysis.md` | Created | Comprehensive untilize multi-core analysis |
| `ttnn/ttnn/operations/row_standardize/agent_logs/analyzer_untilize_execution_log.md` | Created | This execution log |
| `ttnn/ttnn/operations/row_standardize/agent_logs/analyzer_untilize_breadcrumbs.jsonl` | Created | Breadcrumb start event (via init script) |

## Final Status
- **Completed**: Yes
- **Output File**: `/localdev/mstaletovic/tt-metal/ttnn/ttnn/operations/row_standardize/untilize_analysis.md`
- **Execution Log**: `/localdev/mstaletovic/tt-metal/ttnn/ttnn/operations/row_standardize/agent_logs/analyzer_untilize_execution_log.md`
- **Issues**: Breadcrumb append script had JSON parsing issues; used this log file as the primary record instead
