# Execution Log: Softmax (W-Small) Analysis for Row Standardize Reference

## Session Info
- **Started**: 2026-02-09
- **Operation**: softmax (general w_small variant)
- **Program Factory Path**: `ttnn/cpp/ttnn/operations/normalization/softmax/device/softmax_program_factory_general_w_small.cpp`
- **Purpose**: Reference analysis for row_standardize operation

## Execution Timeline

### Step 1: Initial Reconnaissance - Read Program Factory
**Action**: Read the specified program factory file
**Command/Tool**: Read `softmax_program_factory_general.cpp`
**Result**: File only contained `override_runtime_arguments` (33 lines). The `create` method was not present.
**Decision**: The general factory is a base class. Need to find the variant-specific factories. Globbed the device directory.

### Step 2: Identify All Softmax Variants
**Action**: Glob for all files in the softmax device directory
**Command/Tool**: Glob `ttnn/cpp/ttnn/operations/normalization/softmax/device/*` and `**/*.cpp`, `**/*.hpp`
**Result**: Found 7 factory variants: w_small, w_large, h_small, h_large, c_large, attention_optimized, attention_optimized_sharded. Found kernel directories under `device/kernels/attention/`.
**Decision**: Focus on `w_small` variant as it implements row-wise softmax (W dimension reduction) which is most relevant for row_standardize.

### Step 3: Read W-Small Factory and Headers
**Action**: Read `softmax_program_factory_general_w_small.cpp`, `.hpp`, `softmax_device_operation.hpp`, `softmax_operation_types.hpp`
**Result**: Found the complete `create()` function (153 lines). Identified kernel paths pointing to `moreh_softmax/device/kernels/`. Identified CB configuration (9 CBs), compile-time/runtime args, and core distribution logic.
**Decision**: The kernel path is `ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/`. Need to read all three kernels.

### Step 4: Discover Kernel Source Location
**Action**: Glob for moreh_softmax kernels
**Command/Tool**: Glob `ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/*`
**Result**: Found 15 kernel files covering all variants (w, h, c, large/small).
**Decision**: Read the three w-variant kernels: `reader_moreh_softmax_w.cpp`, `moreh_softmax_w.cpp`, `writer_moreh_softmax_w.cpp`

### Step 5: Read All Three Kernels
**Action**: Read reader, compute, and writer kernels in parallel
**Command/Tool**: Three parallel Read calls
**Result**:
- Reader (55 lines): Uses TensorAccessor, generates scaler and mask tiles, reads Wt tiles per row
- Compute (177 lines): 5-stage pipeline using reduce_helpers_compute.hpp and moreh_common.hpp
- Writer (33 lines): Simple sequential tile write with TensorAccessor
**Decision**: Need to understand the helper libraries used: `reduce_helpers_compute.hpp`, `moreh_common.hpp` (both compute and dataflow versions)

### Step 6: Read Helper Libraries
**Action**: Read `reduce_helpers_compute.hpp`, `kernel/compute/moreh_common.hpp`, `kernel/dataflow/moreh_common.hpp`
**Command/Tool**: Three parallel Read calls
**Result**:
- `reduce_helpers_compute.hpp` (446 lines): Comprehensive reduce library with policies (WaitAndPopPerTile, BulkWaitBulkPop, WaitUpfrontNoPop, NoWaitNoPop), accumulation support, post-reduce operations
- `kernel/compute/moreh_common.hpp` (1193 lines): Helper functions like `sub_bcast_cols_init_short_with_dt`, `mul_bcast_cols_init_short_with_dt`, `mask_tile_to_cb`, `pack_tile_with_dt` with FP32_DEST_ACC_EN conditionals
- `kernel/dataflow/moreh_common.hpp` (836 lines): `generate_bcast_scaler`, `generate_mask_w`, `noc_async_read_tile_helper` etc.
**Decision**: Have enough information about helper libraries. Need to understand program factory selection and L1 budget.

### Step 7: DeepWiki Query - Softmax Pipeline
**Action**: Asked DeepWiki about the moreh softmax W-small compute pipeline
**Result**: Confirmed 5-stage pipeline, CB assignments, and that w_small is selected based on L1 memory availability.
**Decision**: Understanding confirmed. Proceed to understand broadcast semantics.

### Step 8: DeepWiki Query - Broadcast Operations
**Action**: Asked DeepWiki about BroadcastType::COL and BroadcastType::ROW
**Result**: COL broadcast treats operand B as column vector, replicating B[h] across all w positions. Hardware unpack stage handles replication. Key for applying per-row statistics back to full tiles.
**Decision**: This is the critical pattern for row_standardize (applying mean/variance per-row).

### Step 9: Read Program Factory Selection and L1 Budget
**Action**: Read `softmax_device_operation.cpp`
**Result**: Found `select_program_factory` logic and `is_softmax_general_w_small_available` function. W-small is selected when dim==rank-1 and CB memory fits in 512KB L1.
**Decision**: Have complete picture of when and how the variant is selected.

### Step 10: Read Helper Function Signatures
**Action**: Grep for `CreateCircularBuffer`, `CreateComputeKernel`, `ComputeKernelArg`, `CircularBufferArg` in moreh_helper_functions.hpp
**Result**: Understood the helper API structures used by the factory.
**Decision**: Have all information needed to write the analysis.

## Files Read
| File | Purpose | Key Findings |
|------|---------|--------------|
| `softmax_program_factory_general.cpp` | Initial target file | Only contains override_runtime_arguments; create() is in variant subclasses |
| `softmax_program_factory_general.hpp` | Base class header | Defines shared_variables_t with reader/writer kernel IDs and core count |
| `softmax_program_factory_general_w_small.cpp` | **Primary target** - W-small factory | Complete create() with 9 CBs, 3 kernels, core distribution, runtime args |
| `softmax_program_factory_general_w_small.hpp` | W-small header | Inherits from SoftmaxProgramFactoryGeneral |
| `softmax_device_operation.hpp` | Device op header | Shows all 7 factory variants in program_factory_t variant |
| `softmax_operation_types.hpp` | Type definitions | SoftmaxParams, SoftmaxInputs, kernel path constants |
| `softmax_device_operation.cpp` | Factory selection and validation | L1 budget check, select_program_factory logic |
| `reader_moreh_softmax_w.cpp` | Reader kernel | TensorAccessor, scaler/mask generation, Wt-tile row reads |
| `moreh_softmax_w.cpp` | **Compute kernel** | 5-stage pipeline: MAX, SUB, EXP, SUM+RECIP, MUL |
| `writer_moreh_softmax_w.cpp` | Writer kernel | Simple sequential tile writes |
| `reduce_helpers_compute.hpp` | Reduce library | Policies, accumulation, post-reduce ops |
| `kernel/compute/moreh_common.hpp` | Compute helpers | _with_dt variants, mask_tile_to_cb, broadcast helpers |
| `kernel/dataflow/moreh_common.hpp` | Dataflow helpers | generate_bcast_scaler, generate_mask_w |
| `moreh_helper_functions.hpp` | Factory helpers | CircularBufferArg, ComputeKernelArg, split_work_to_cores |

## DeepWiki Queries
| Query | Response Summary | How Used |
|-------|------------------|----------|
| Softmax W-small compute pipeline | 5-stage pipeline confirmed, CB usage documented | Validated understanding of pipeline stages |
| BroadcastType::COL and ROW semantics | COL broadcasts column vector across rows, ROW broadcasts row vector across columns | Critical for understanding how per-row reduction results are applied back to full tiles |

## Key Decisions
| Decision Point | Options Considered | Choice Made | Rationale |
|----------------|-------------------|-------------|-----------|
| Which variant to analyze | w_small, w_large, h_small, h_large, c_large, attention | w_small | Row_standardize operates on last dim (W), and w_small is the simpler case that demonstrates the core pattern |
| How deep to go on moreh helpers | Just signatures vs full implementation | Signatures + key function bodies | Enough to understand the pattern without getting lost in implementation details |
| Include LOG variant details | Skip LOG, detail LOG, mention briefly | Mention briefly | LOG is an alternative mode that demonstrates code flexibility but is not used for standard softmax |

## Files Created/Modified
| File | Action | Description |
|------|--------|-------------|
| `ttnn/ttnn/operations/row_standardize/softmax_analysis.md` | Created | Comprehensive analysis of softmax w_small variant as reference for row_standardize |
| `ttnn/ttnn/operations/row_standardize/agent_logs/analyzer_softmax_execution_log.md` | Created | This execution log |
| `ttnn/ttnn/operations/row_standardize/agent_logs/analyzer_softmax_breadcrumbs.jsonl` | Created | Breadcrumb tracking file |

## Final Status
- **Completed**: Yes
- **Output File**: `/localdev/mstaletovic/tt-metal/ttnn/ttnn/operations/row_standardize/softmax_analysis.md`
- **Issues**: None. All sources were accessible and consistent.
