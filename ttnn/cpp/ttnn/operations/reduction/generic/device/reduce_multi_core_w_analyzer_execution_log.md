# Execution Log: Reduce Multi-Core W Analysis

## Session Info
- **Started**: Session start
- **Operation**: reduce_multi_core_w
- **Program Factory Path**: `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_op_multi_core_w_program_factory.cpp`
- **Analysis Purpose**: Reference for creating hybrid variance operation (mean, subtract, square, mean-of-squares along W dimension)

## Execution Timeline

### Step 1: Read Program Factory and Table Templates
**Action**: Read the main program factory file and reference table templates
**Command/Tool**: Read tool on two files in parallel
**Files Read**:
- `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_op_multi_core_w_program_factory.cpp`
- `/localdev/mstaletovic/tt-metal/.claude/references/table-templates.md`
**Result**: Successfully read both files
**Key Findings**:
- Program factory uses `split_work_to_cores` for work distribution across NC*Ht rows
- Three circular buffers: c_0 (input), c_2 (scaler), c_3 (output)
- Two kernel paths: one for generic reduce, one for matmul-based SUM reduction
- Scaler is packed as two bfloat16 values into uint32
- TensorAccessorArgs used for compile-time buffer configuration

### Step 2: Read All Referenced Kernel Files
**Action**: Read the three kernel files referenced in the program factory
**Command/Tool**: Read tool on three files in parallel
**Files Read**:
- `ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/reader_unary_reduce_universal_start_id.cpp`
- `ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce_w.cpp`
- `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_universal_start_id.cpp`
**Result**: All three files read successfully
**Key Findings**:
- Reader generates scaler tile at startup using `generate_reduce_scaler()`
- Compute kernel has two paths: `#ifndef REDUCE_ROW_SUM_VIA_MM` (reduce_helpers library) vs matmul-based
- Writer is generic unary writer with TensorAccessor
- STREAMING mode in reduce_helpers processes one tile at a time

### Step 3: Read reduce_helpers.hpp Library
**Action**: Read the compute kernel library for reduction operations
**Command/Tool**: Read tool
**File Read**: `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/kernel_lib/reduce_helpers.hpp`
**Result**: Successfully read (738 lines)
**Key Findings**:
- Unified reduce function with template parameters for reduce_type, reduce_dim, input_mode
- Four input modes: STREAMING, STREAMING_BATCHED, PRELOADED, PERSISTENT
- REDUCE_ROW pattern: For each row (Ht iterations), reduces Wt tiles to 1 output tile
- Uses tile_regs_acquire/commit/wait/release for DST management
- Supports accumulation via Accumulate type parameter

### Step 4: Find and Read generate_reduce_scaler.hpp
**Action**: Locate the scaler generation utility
**Command/Tool**: Glob for `**/generate_reduce_scaler.hpp`, then Read
**File Found**: `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp`
**Result**: Successfully read
**Key Findings**:
- `generate_reduce_scaler()` creates a tile with scaler values in specific positions
- First fills tile with zeros via NOC read from MEM_ZEROS_BASE
- Then sets first 8 elements of each face (4 faces total) to the packed scaler value
- Scaler is double-packed bfloat16 in uint32 format

### Step 5: DeepWiki Query - reduce_tile Function
**Action**: Query DeepWiki about reduce_tile mechanics
**Query**: "How does the reduce_tile function work in compute kernels? What are the parameters and how does it perform row reduction (REDUCE_ROW) for the W dimension?"
**Result**: Received comprehensive response
**Key Findings**:
- reduce_tile takes icb (input CB), icb_scaler (scaler CB), itile (input index), itile_scaler (scaler index), idst (DST index)
- REDUCE_ROW collapses columns into single column (W dimension reduction)
- Requires reduce_init before use and reduce_uninit after
- Template parameters: reduce_type (SUM/AVG/MAX), reduce_dim (REDUCE_ROW/COL/SCALAR)

### Step 6: DeepWiki Query - REDUCE_ROW_SUM_VIA_MM
**Action**: Query about the matmul-based optimization
**Query**: "What is the REDUCE_ROW_SUM_VIA_MM approach for row reduction?"
**Result**: Received detailed explanation
**Key Findings**:
- REDUCE_ROW_SUM_VIA_MM uses matrix multiplication instead of reduce_tile for SUM operations
- Leverages hardware's optimized matrix multiply units
- Enabled via define when reduce_dim==W and reduce_op==SUM
- matmul_tiles accumulates results into DST register

### Step 7: DeepWiki Query - TensorAccessor
**Action**: Query about memory access abstraction
**Query**: "How does TensorAccessor work in tt-metal dataflow kernels?"
**Result**: Comprehensive response received
**Key Findings**:
- TensorAccessor abstracts memory bank distribution for efficient access
- TensorAccessorArgs configures compile-time vs runtime parameter passing
- noc_async_read_page uses TensorAccessor to translate logical page ID to physical address
- Supports interleaved and sharded memory layouts

### Step 8: DeepWiki Query - split_work_to_cores
**Action**: Query about work distribution mechanism
**Query**: "How does split_work_to_cores work in tt-metal?"
**Result**: Detailed explanation received
**Key Findings**:
- Returns two core groups: core_group_1 (more work) and core_group_2 (less work)
- Handles uneven division by assigning +1 unit to group_1 cores
- Returns num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2

### Step 9: DeepWiki Query - Tile Layout
**Action**: Query about tile data format
**Query**: "What is the tile layout and data format in Tenstorrent hardware?"
**Result**: Comprehensive tile structure information
**Key Findings**:
- 32x32 element tiles divided into four 16x16 faces
- bfloat16 tile = 2048 bytes
- Faces stored row-major: face0->face1->face2->face3
- Matrix engine natively multiplies 16x16 matrices

### Step 10: Read reduce_op.cpp for Define Generation
**Action**: Understand how defines are generated
**Command/Tool**: Read tool
**File Read**: `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_op.cpp`
**Result**: Successfully read
**Key Findings**:
- `get_defines()` maps ReduceOpDim to ReduceDim strings
- W -> REDUCE_ROW, H -> REDUCE_COL, HW -> REDUCE_SCALAR
- REDUCE_ROW_SUM_VIA_MM set to "1" only for SUM + W combination
- Multi-core HW reduction chains W reduction then H reduction

### Step 11: Read Supporting Headers
**Action**: Read operation types and factory header
**Command/Tool**: Read tool on multiple files
**Files Read**:
- `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_op_multi_core_w_program_factory.hpp`
- `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_op_device_operation_types.hpp`
- `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/reduction/generic/device/common.hpp`
- `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp`
**Result**: All read successfully
**Key Findings**:
- shared_variables_t stores reader_kernel_id, writer_kernel_id, and cores vector
- operation_attributes_t contains math_op, dim, scaler, output_mem_config, compute_kernel_config
- ReduceOpMath: SUM, MAX, MIN
- ReduceOpDim: H, W, HW
- DEST_AUTO_LIMIT from dest_helpers.hpp provides compile-time DEST capacity

### Step 12: Write Analysis Document
**Action**: Create comprehensive analysis markdown file
**Command/Tool**: Write tool
**File Created**: `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_multi_core_w_analysis.md`
**Result**: Successfully created
**Content**: Full analysis with all sections as specified in agent definition

### Step 13: Write Execution Log
**Action**: Create this execution log file
**Command/Tool**: Write tool
**File Created**: `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_multi_core_w_analyzer_execution_log.md`

## Files Read

| File | Purpose | Key Findings |
|------|---------|--------------|
| reduce_op_multi_core_w_program_factory.cpp | Main program factory | CB setup, work distribution, kernel creation |
| reader_unary_reduce_universal_start_id.cpp | Reader kernel | Scaler generation, tile reading loop |
| reduce_w.cpp | Compute kernel | Two paths: reduce_helpers vs matmul |
| writer_unary_universal_start_id.cpp | Writer kernel | Generic unary writer with TensorAccessor |
| reduce_helpers.hpp | Compute library | Unified reduce with modes, DST management |
| generate_reduce_scaler.hpp | Scaler utility | Tile generation with packed bfloat16 |
| reduce_op.cpp | Define generation | REDUCE_ROW_SUM_VIA_MM logic |
| reduce_op_multi_core_w_program_factory.hpp | Factory header | shared_variables_t structure |
| reduce_op_device_operation_types.hpp | Operation types | operation_attributes_t, tensor_args_t |
| common.hpp | Enums | ReduceOpMath, ReduceOpDim, ReduceOpParallelizationStrategy |
| dest_helpers.hpp | DEST utilities | DEST_AUTO_LIMIT, get_dest_limit() |
| table-templates.md | Output format | Standard table formats for analysis |

## DeepWiki Queries

| Query | Response Summary | How Used |
|-------|------------------|----------|
| reduce_tile function | Parameters: icb, icb_scaler, itile, itile_scaler, idst. REDUCE_ROW collapses W dimension | Documented in compute kernel analysis |
| REDUCE_ROW_SUM_VIA_MM | Uses matmul instead of reduce_tile for SUM, leverages HW MM units | Explained optimization in Implementation Notes |
| TensorAccessor | Abstracts memory bank distribution, noc_async_read_page uses it | Documented in Index Calculations section |
| split_work_to_cores | Returns two core groups with different work amounts | Documented in Core Distribution Strategy |
| Tile layout | 32x32 tiles, 4 faces of 16x16, bfloat16=2048 bytes | Used for CB sizing documentation |

## Errors Encountered

| Error | Context | Resolution |
|-------|---------|------------|
| File not found for generate_reduce_scaler.hpp | Initial path was incorrect | Used Glob to find correct path in deprecated directory |

## Key Decisions

| Decision Point | Options Considered | Choice Made | Rationale |
|----------------|-------------------|-------------|-----------|
| Analysis focus areas | Full analysis vs targeted | Targeted focus on W reduction, scaler CB, CB flow per user request | User specified focus on variance computation requirements |
| REDUCE_ROW_SUM_VIA_MM documentation | Brief mention vs detailed | Detailed explanation | Critical for understanding SUM performance optimization |
| Scaler tile documentation | Skip vs document structure | Document structure | Essential for understanding mean computation (1/W scaler) |

## Files Created/Modified

| File | Action | Description |
|------|--------|-------------|
| reduce_multi_core_w_analysis.md | Created | Comprehensive analysis document |
| reduce_multi_core_w_analyzer_execution_log.md | Created | This execution log |

## Final Status
- **Completed**: Yes
- **Output File**: `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_multi_core_w_analysis.md`
- **Execution Log**: `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_multi_core_w_analyzer_execution_log.md`
- **Issues**: None - all sections completed successfully
