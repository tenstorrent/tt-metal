# Row Standardize — Pipeline Report

## 1. Executive Summary

**Operation**: `row_standardize` — per-row standardization (layer norm without gamma/beta) on Tenstorrent hardware.

**Formula**: `output[..., i] = (x[..., i] - mean_row) * rsqrt(var_row + epsilon)`

**What was built**:
- A complete Python-based TTNN operation using the generic_op infrastructure
- 3 RISC-V kernels (reader, compute, writer) implementing a tilize → 6-phase row standardize → untilize pipeline
- Comprehensive test suite with 29 tests (22 correctness + 2 minimal + 4 validation + 1 edge case)

**Final test status**: **ALL 29 TESTS PASS**
- 11 bfloat16 shapes: PCC > 0.99
- 11 float32 shapes: PCC > 0.999
- Validation and edge case tests: all pass

**Supported dtypes**: bfloat16, float32
**Supported shapes**: Any tensor with rank >= 2, H and W multiples of 32
**Layout**: ROW_MAJOR input → ROW_MAJOR output (internal tilize/untilize in compute kernel)

---

## 2. Pipeline Overview

```
Phase 1: Analyzers (parallel)    Phase 2: Planner    Phase 3: Parallel Build    Phase 4: Kernel Writer
┌─────────────────────┐
│ analyzer_tilize      │──┐
├─────────────────────┤  │    ┌──────────┐    ┌──────────────────────┐    ┌───────────────┐
│ analyzer_softmax     │──┼───>│ planner  │───>│ generic_op_builder   │───>│ kernel_writer │
├─────────────────────┤  │    └──────────┘    ├──────────────────────┤    └───────────────┘
│ analyzer_untilize    │──┘                   │ kernel_designer      │───>
└─────────────────────┘                       └──────────────────────┘
```

| Phase | Agent(s) | Duration | Status |
|-------|----------|----------|--------|
| Phase 1 | analyzer_tilize, analyzer_softmax, analyzer_untilize (parallel) | ~7.5 min | Complete |
| Phase 2 | planner | ~8.4 min | Complete |
| Phase 3a | generic_op_builder | ~7.2 min | Complete |
| Phase 3b | kernel_designer (parallel with 3a) | ~6.6 min | Complete |
| Phase 4 | kernel_writer | ~11.2 min | Complete |
| **Total** | | **~34.3 min** | **Success** |

---

## 3. Per-Agent Summaries

### 3.1 Analyzer: Tilize (input_stage)

**Reference**: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_multi_core_interleaved_program_factory.cpp`

**What it produced**: `tilize_analysis.md` — comprehensive analysis of multi-core interleaved tilize operation.

**Key findings**:
- Reader reads 32 RM sticks per block using TensorAccessor + noc_async_read
- Tilize performed in compute kernel using hardware-accelerated `tilize_block` / `compute_kernel_lib::tilize<>()`
- CB page sizes use tile_size even for RM sticks (synchronization granularity)
- Multi-core work distribution via `split_blocks_for_tilize(grid, num_tile_rows)`

**Decisions**: None required (pure analysis).

**Logs**: `agent_logs/analyzer_tilize_execution_log.md`

---

### 3.2 Analyzer: Softmax (compute_core)

**Reference**: `ttnn/cpp/ttnn/operations/normalization/softmax/device/softmax_program_factory_general.cpp`

**What it produced**: `softmax_analysis.md` — detailed analysis of general softmax (W_small variant).

**Key findings**:
- 5-phase compute pipeline: MAX reduce → sub(x-max) → exp → SUM reduce + recip → mul
- Maps directly to row_standardize with modifications (SUM instead of MAX, square+rsqrt instead of exp+recip)
- Uses `WaitUpfrontNoPop` policy to reuse tiles across pipeline stages
- `BroadcastType::COL` for applying per-row scalars back to full tiles
- Scaler tile generation via `generate_bcast_scaler` (always bfloat16 format)
- 9 CBs with indices c_0 through c_28

**Decisions**: None required (pure analysis).

**Logs**: `agent_logs/analyzer_softmax_execution_log.md`, `agent_logs/analyzer_softmax_breadcrumbs.jsonl`

---

### 3.3 Analyzer: Untilize (output_stage)

**Reference**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/factories/untilize_multi_core_program_factory.cpp`

**What it produced**: `untilize_analysis.md` — comprehensive analysis of multi-core untilize operation.

**Key findings**:
- Untilize performed in compute kernel using hardware-accelerated `pack_untilize`/`untilize`
- Writer extracts individual rows from untilized CB data and writes them row-by-row to DRAM
- Output page size in CB is still tile-sized even though data is row-major
- Multi-core distribution via `split_blocks_for_tilize(grid, num_tiles_per_col)`

**Decisions**: None required (pure analysis).

**Logs**: `agent_logs/analyzer_untilize_execution_log.md`, `agent_logs/analyzer_untilize_breadcrumbs.jsonl`

---

### 3.4 Planner

**What it produced**: `row_standardize_spec.md` — functional specification for the operation.

**Key decisions**:
1. **Single compute kernel** — tilize + row_standardize + untilize all in one kernel to avoid DRAM round-trips
2. **Single-core prototype** — simplifies implementation; multi-core extension is straightforward
3. **Row-wise processing** — one full tile-row at a time (matching softmax w_small pattern)
4. **Reduce scaler = 1/W** — embedded in hardware reduce to directly compute mean/variance
5. **Epsilon as scalar broadcast tile** — generated once by reader, used via `add_tiles_bcast_scalar`
6. **11 circular buffers** — c_0 through c_28 covering input, output, scalars, and 5 intermediates
7. **Dtype-aware CB sizing** — 2048 bytes/tile for bf16, 4096 bytes/tile for f32

**Pain points**: None significant.

**Logs**: `agent_logs/planner_execution_log.md`, `agent_logs/planner_breadcrumbs.jsonl`

---

### 3.5 Generic Op Builder

**What it produced**:
- `__init__.py` — package re-export
- `row_standardize.py` — entry point with validation and output allocation
- `row_standardize_program_descriptor.py` — ProgramDescriptor with 11 CBs, 3 kernels, runtime args
- `test_row_standardize.py` — 29 pytest cases (shapes x dtypes + validation + edge case)
- Stub kernels (3 `.cpp` files)

**Key decisions**:
1. CB page size = `tile.get_tile_size(dtype)` for all CBs (including RM CBs)
2. Intermediate format: Float32 if fp32_dest_acc_en, else same as input dtype
3. Bfloat16 packing via `_float_to_bfloat16()` helper (truncate lower 16 bits)
4. Wt and nblocks as compile-time args for compute kernel
5. Empty RuntimeArgs for compute kernel (single-core, all work from CT args)

**Pain points**:
- `ttnn.Shape` objects cannot be sliced with standard Python slicing
- Kernel path format unclear (absolute vs relative) — used absolute

**Logs**: `agent_logs/generic_op_builder_execution_log.md`, `agent_logs/generic_op_builder_breadcrumbs.jsonl`

---

### 3.6 Kernel Designer

**What it produced**: `kernel_design.md` — per-phase helper/raw-call decisions for all 3 kernels.

**Key decisions**:
1. **Phase 6 (add eps + rsqrt) is the only NO HELPER phase** — requires raw tile API calls with manual DST management
2. **cb_xmm reuse strategy**: Phase 4 uses `WaitUpfrontNoPop` to keep tiles alive for Phase 7
3. **Phase 3 sub**: `WaitUpfrontPopAtEnd` for A (frees tilized), `WaitAndPopPerTile` for B (frees mean)
4. **Phase 7 mul**: `WaitUpfrontPopAtEnd` for A (frees xmm), `WaitAndPopPerTile` for B (frees invstd)

**Pain points**:
- Spec's cb_eps format description was ambiguous for SCALAR broadcast (resolved: use `generate_bcast_scalar_bfloat16`)
- Wt must be `constexpr` for untilize template parameter (noted as critical constraint)

**Logs**: `agent_logs/kernel_designer_execution_log.md`, `agent_logs/ttnn-kernel-designer_breadcrumbs.jsonl`

---

### 3.7 Kernel Writer

**What it produced**: Working implementations of all 3 kernels:
- `kernels/row_standardize_reader.cpp` (95 lines)
- `kernels/row_standardize_compute.cpp` (162 lines)
- `kernels/row_standardize_writer.cpp` (63 lines)

**Key decisions**:
1. `compute_kernel_hw_startup(c_0, c_1, c_16)` — chose first input, scaler, output CBs for startup
2. Reader reads full `stick_size_bytes` per row (standard tilize pattern)

**Pain points**:
1. **Scaler CB format mismatch for float32** — Design doc specified `dtype` format for cb_scaler, but `generate_reduce_scaler` ALWAYS writes packed bf16 values. Using float32 format for the scaler CB caused PCC=0.69. **Fix**: Always use bfloat16 format for cb_scaler.
2. **`ttnn.DataFormat` API does not exist** — Program descriptor originally used `ttnn.DataFormat.Float16_b`. **Fix**: Changed to `ttnn.bfloat16` (DataType enum).
3. **Non-existent `sfpu_split_work.h` header** — Only `compute_kernel_api/eltwise_unary/rsqrt.h` was needed.

**Deviations from design**:

| Deviation | Justification |
|-----------|---------------|
| cb_scaler (c_1) uses bfloat16 instead of `dtype` | `generate_reduce_scaler` always writes packed bf16. Softmax reference confirms scaler CB is always `Float16_b`. Without this fix, float32 produces PCC=0.69. |
| Fixed program descriptor DataFormat -> DataType | `ttnn.DataFormat` does not exist in Python API; `CBFormatDescriptor` accepts `DataType` enum values |
| Scaler always packed as bf16 (not float32 for f32 input) | `generate_reduce_scaler` asserts `(scaler >> 16) == (scaler & 0xFFFF)`, enforcing packed bf16 format |

**Logs**: `agent_logs/kernel_writer_execution_log.md`, `agent_logs/kernel_writer_breadcrumbs.jsonl`

---

## 4. Architecture Summary

### Data Flow
```
DRAM (RM sticks) → Reader → cb_rm_in (c_0)
                                 ↓
                    Compute: tilize → cb_tilized (c_3)
                                 ↓
                    Compute: SUM reduce * 1/W → cb_mean (c_24)
                                 ↓
                    Compute: sub<COL> → cb_xmm (c_25)  [tiles persist for Phase 7]
                                 ↓
                    Compute: square → cb_xmm_sq (c_26)
                                 ↓
                    Compute: SUM reduce * 1/W → cb_var (c_27)
                                 ↓
                    Compute: add eps + rsqrt → cb_invstd (c_28)
                                 ↓
                    Compute: mul<COL> (cb_xmm * cb_invstd) → cb_tilized_out (c_4)
                                 ↓
                    Compute: untilize → cb_rm_out (c_16)
                                 ↓
                    Writer → DRAM (RM sticks)
```

### Circular Buffer Layout

| CB | Name | Pages | Format | Purpose |
|----|------|-------|--------|---------|
| c_0 | cb_rm_in | Wt | dtype | Input RM sticks |
| c_1 | cb_scaler | 1 | bfloat16 (always) | Reduce scaler 1/W |
| c_2 | cb_eps | 1 | intermed_fmt | Epsilon scalar |
| c_3 | cb_tilized | Wt | intermed_fmt | Tilized input |
| c_4 | cb_tilized_out | Wt | intermed_fmt | Normalized tiles |
| c_16 | cb_rm_out | Wt | dtype | Output RM sticks |
| c_24 | cb_mean | 1 | intermed_fmt | Row means |
| c_25 | cb_xmm | Wt | intermed_fmt | x - mean |
| c_26 | cb_xmm_sq | Wt | intermed_fmt | (x - mean)^2 |
| c_27 | cb_var | 1 | intermed_fmt | Row variance |
| c_28 | cb_invstd | 1 | intermed_fmt | rsqrt(var + eps) |

**L1 Budget**: ~(5*Wt + 5) * tile_size. For W=1024 (Wt=32) with bf16: ~330 KB (fits L1).

### Kernel Structure

| Kernel | Core | Phases |
|--------|------|--------|
| Reader (BRISC) | NOC0 | 1. Generate scaler tile (once), 2. Generate epsilon tile (once), 3. Read 32 RM sticks per block |
| Compute (TRISC) | FPU/SFPU | 8 phases per block: tilize → mean → sub → square → var → add+rsqrt → mul → untilize |
| Writer (NCRISC) | NOC1 | Write 32 RM sticks per block row-by-row |

---

## 5. Test Results

### Correctness Tests (22 tests: 11 shapes x 2 dtypes)

All tests pass with PCC well above thresholds.

| Shape | bfloat16 (PCC > 0.99) | float32 (PCC > 0.999) |
|-------|----------------------|----------------------|
| (32, 32) | PASS | PASS |
| (32, 64) | PASS | PASS |
| (64, 128) | PASS | PASS |
| (128, 128) | PASS | PASS |
| (32, 1024) | PASS | PASS |
| (128, 1024) | PASS | PASS |
| (1024, 32) | PASS | PASS |
| (1024, 1024) | PASS | PASS |
| (2, 32, 64) | PASS | PASS |
| (4, 64, 128) | PASS | PASS |
| (2, 4, 32, 64) | PASS | PASS |

### Other Tests (7 tests)

| Test | Status |
|------|--------|
| test_row_standardize_minimal[bf16] | PASS |
| test_row_standardize_minimal[f32] | PASS |
| test_row_standardize_validation_rank | PASS |
| test_row_standardize_validation_layout | PASS |
| test_row_standardize_validation_dtype | PASS |
| test_row_standardize_validation_width_alignment | PASS |
| test_row_standardize_constant_row | PASS |

---

## 6. Lessons Learned

### Cross-Agent Issues

1. **Scaler CB format is ALWAYS bfloat16** — This was the biggest cross-agent issue. The spec and design doc specified the scaler CB format as `dtype` (matching input), but `generate_reduce_scaler` ALWAYS writes packed bfloat16 values. The kernel writer discovered this when float32 tests produced PCC=0.69 and had to fix the program descriptor. **Recommendation**: The planner/designer should explicitly document that reduce scaler CBs must always be bfloat16 regardless of input dtype.

2. **`ttnn.DataFormat` vs `ttnn.DataType`** — The generic_op_builder used a non-existent API (`ttnn.DataFormat.Float16_b`). The kernel writer had to fix this to `ttnn.bfloat16`. **Recommendation**: The generic_op_builder should validate API existence against the actual Python module.

3. **Parallel agent file commits** — Parallel agents (generic_op_builder and kernel_designer) occasionally committed each other's files, causing merge conflicts in git. This is a known issue with the parallel execution model.

### What Worked Well

1. **Hybrid mode composition** — The 3-reference approach (tilize + softmax + untilize) provided a clear template for each stage. The softmax compute pipeline mapped almost directly to row standardize.

2. **Kernel helper library** — The `reduce_helpers_compute.hpp` and `binary_op_helpers.hpp` libraries handled 7 of 8 compute phases. Only Phase 6 (add eps + rsqrt) required raw tile API calls.

3. **Policy-based CB management** — The `WaitUpfrontNoPop`, `WaitUpfrontPopAtEnd`, `WaitAndPopPerTile`, and `BulkWaitBulkPop` policies enabled precise control over CB lifetimes without manual push/pop orchestration.

4. **Single-core prototype** — Starting with single-core avoided complexity while proving correctness. Multi-core extension is straightforward since each tile-row is independent.

### Potential Improvements

1. **Multi-core support** — Add `split_blocks_for_tilize` distribution for performance on large tensors
2. **Large-W variant** — For W > ~1568 (bf16) or W > ~768 (f32), tiles exceed L1 budget
3. **Sharded memory** — Support sharded input/output for better data locality
4. **Optional gamma/beta** — Extend to full layer norm with learnable parameters

---

## 7. File Inventory

```
ttnn/ttnn/operations/row_standardize/
├── __init__.py                           # Package re-export
├── row_standardize.py                    # Entry point + validation
├── row_standardize_program_descriptor.py # ProgramDescriptor (11 CBs, 3 kernels)
├── test_row_standardize.py               # 29 pytest cases
├── kernels/
│   ├── row_standardize_reader.cpp        # Reader: sticks + scaler/eps generation
│   ├── row_standardize_compute.cpp       # 8-phase compute pipeline
│   └── row_standardize_writer.cpp        # Writer: sticks to DRAM
├── row_standardize_spec.md               # Functional specification
├── kernel_design.md                      # Kernel implementation design
├── tilize_analysis.md                    # Tilize reference analysis
├── softmax_analysis.md                   # Softmax reference analysis
├── untilize_analysis.md                  # Untilize reference analysis
├── pipeline_report.md                    # This report
├── IMPLEMENTATION_SUMMARY.md             # Generic op builder summary
└── agent_logs/
    ├── logging_config.json
    ├── analyzer_tilize_execution_log.md
    ├── analyzer_softmax_execution_log.md
    ├── analyzer_softmax_breadcrumbs.jsonl
    ├── analyzer_untilize_execution_log.md
    ├── analyzer_untilize_breadcrumbs.jsonl
    ├── planner_execution_log.md
    ├── planner_breadcrumbs.jsonl
    ├── generic_op_builder_execution_log.md
    ├── generic_op_builder_breadcrumbs.jsonl
    ├── kernel_designer_execution_log.md
    ├── ttnn-kernel-designer_breadcrumbs.jsonl
    ├── kernel_writer_execution_log.md
    └── kernel_writer_breadcrumbs.jsonl
```
