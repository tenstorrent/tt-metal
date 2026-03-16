# Pipeline Report: layer_norm_rm

## Summary

**Operation**: `layer_norm_rm` — Per-row layer normalization on row-major interleaved tensors
**Result**: ALL 4 TDD STAGES PASSED
**Total pipeline duration**: ~60 minutes (wall clock)
**Branch**: `2026_03_16_0855_run1_layer_norm_rm`

### What it does
Computes `output = (x - mean) * rsqrt(var + eps) * gamma + beta` per row, where mean and variance are computed along the last dimension. Input/output are ROW_MAJOR_LAYOUT bfloat16 tensors. Optional gamma (scale) and beta (shift) affine parameters.

### API
```python
from ttnn.operations.layer_norm_rm import layer_norm_rm

output = layer_norm_rm(input_tensor, gamma=None, beta=None, *, epsilon=1e-5)
```

---

## Pipeline Execution

| Phase | Agent | Duration | Output |
|-------|-------|----------|--------|
| 0: Discovery | orchestrator | ~1 min | Identified tilize, untilize, batch_norm as references |
| 1: Analysis | 3× ttnn-operation-analyzer (parallel) | ~12 min | tilize_analysis.md, untilize_analysis.md, batch_norm_analysis.md |
| 2: Design | ttnn-operation-architect | ~11 min | op_design.md, .tdd_state.json (4 stages) |
| 3: Build | ttnn-generic-op-builder | ~10 min | Python infra, stub kernels, TDD stage tests |
| 4: TDD Kernels | ttnn-kernel-writer-tdd | ~21 min | All 3 kernel files implemented, 4/4 stages passed |
| 5: Report | orchestrator | ~1 min | This file |

---

## Agent Summaries

### Phase 1: Analyzers (3 parallel agents)

**Tilize analyzer** (input_stage):
- Analyzed `tilize_multi_core_default_program_factory.cpp`
- Key findings: Reader reads RM sticks via NoC, compute uses `tilize_block(icb, block, ocb)`, stick_size = W × element_size, work unit is tile-row (32 sticks)

**Untilize analyzer** (output_stage):
- Analyzed `untilize_multi_core_program_factory.cpp`
- Key findings: `untilize_block(icb, full_ct_dim, ocb)` in compute, writer uses TensorAccessor for RM stick writes, one `noc_async_write_barrier` per block

**Batch norm analyzer** (compute_core):
- Analyzed `batch_norm_program_factory.cpp`
- Key findings: `tile_regs_acquire/commit/wait/release` pipeline, `binary_dest_reuse_tiles` for chained ops, per-channel broadcast reuse pattern applicable to per-row pattern

### Phase 2: Architect

- Designed 9-phase compute pipeline: tilize → reduce(mean) → sub(COL) → square → reduce(var) → add(eps)+rsqrt → mul(COL) → affine → untilize
- 12 circular buffers defined (cb_in through cb_norm)
- Single-core (0,0) work distribution
- 4 TDD stages registered: data_pipeline, centering, normalize, affine

### Phase 3: Builder

- Created Python entry point with full validation
- Created program descriptor with 12 CBs, 3 kernels, packed bf16 scalers
- Created stub kernels and 4 TDD stage test files + integration test
- Fixed: conftest.py needed `device` fixture

### Phase 4: TDD Kernel Writer

- Implemented all 3 kernels across 4 incremental stages
- All stages passed on first attempt after initial infrastructure fixes

---

## TDD Pipeline Results

| Stage | Name | Result | Attempts | Test File |
|-------|------|--------|----------|-----------|
| 1 | data_pipeline | PASS | 1 (+ infrastructure fixes) | test_stage_data_pipeline.py |
| 2 | centering | PASS | 1 | test_stage_centering.py |
| 3 | normalize | PASS | 1 | test_stage_normalize.py |
| 4 | affine | PASS | 1 | test_stage_affine.py |

### Test Shapes Validated
- `(1, 1, 32, 32)` — minimal single-tile
- `(1, 1, 64, 128)` — multi-tile width
- `(1, 1, 32, 256)` — non-square
- `(4, 2, 64, 64)` — multi-batch

### Upstream Fixes During TDD
1. **conftest.py**: `ttnn.open_device(0)` → `ttnn.open_device(device_id=0)` (keyword arg required)
2. **Reader kernel**: `reinterpret_cast` strict-aliasing violation → union-based float conversion
3. **Program descriptor**: TensorAccessorArgs placeholder `[0]` → `[0, 0]` (interleaved needs 2 CT args)
4. **Program descriptor**: Scaler/eps from packed bf16 → raw float32 bit patterns (generate_reduce_scaler expects float bits)

---

## Files Produced

### Operation directory (`ttnn/ttnn/operations/layer_norm_rm/`)
```
├── __init__.py                          # Re-exports layer_norm_rm
├── layer_norm_rm.py                     # Entry point with validation
├── layer_norm_rm_program_descriptor.py  # CB config, kernel setup, runtime args
├── kernels/
│   ├── layer_norm_rm_reader.cpp         # RM stick reader + scaler/gamma/beta loading
│   ├── layer_norm_rm_compute.cpp        # 9-phase compute: tilize→normalize→untilize
│   └── layer_norm_rm_writer.cpp         # RM stick writer via TensorAccessor
├── op_design.md                         # Architecture + kernel design document
├── .tdd_state.json                      # TDD pipeline state (all passed)
├── REPORT.md                            # This file
└── agent_logs/                          # Breadcrumb files for all agents
```

### Test directory (`tests/ttnn/unit_tests/operations/layer_norm_rm/`)
```
├── conftest.py                          # Device fixture
├── test_stage_data_pipeline.py          # Stage 1: identity passthrough
├── test_stage_centering.py              # Stage 2: x - mean
├── test_stage_normalize.py              # Stage 3: full normalization
├── test_stage_affine.py                 # Stage 4: gamma/beta affine
└── test_layer_norm_rm.py                # Integration test
```

---

## Git History

```
d2921b5 [ttnn-kernel-writer-tdd] stage affine: passed
7f2ff1e [ttnn-kernel-writer-tdd] stage normalize: passed
301fa7b [ttnn-kernel-writer-tdd] stage centering: passed
9a62d98 [ttnn-kernel-writer-tdd] stage data_pipeline: passed
8ce0965 [ttnn-generic-op-builder] add execution log and final breadcrumbs
b1aecb8 [ttnn-generic-op-builder] stubs: layer_norm_rm
524dd7d [ttnn-operation-architect] breadcrumbs: finalize completion log
aa12b88 [ttnn-operation-architect] design: layer_norm_rm
379574c [ttnn-operation-analyzer] analysis: batch_norm (compute_core reference)
67d6d57 [ttnn-operation-analyzer] analysis: untilize (output_stage reference)
```

---

## Decisions and Deviations

### Decisions Made
1. **Single-core**: Used single core (0,0) for simplicity — sufficient for correctness validation
2. **Batch norm as compute reference**: Chosen over softmax (which was deleted from repo) — provided useful patterns for normalization compute
3. **4 TDD stages** (not 6): Architect consolidated stages to avoid intermediate shape issues — data_pipeline, centering, normalize, affine
4. **generate_reduce_scaler for both scalers**: Used for both 1/W and epsilon — works correctly for SCALAR broadcast
5. **Union-based float conversion**: In reader kernel to avoid strict-aliasing warnings

### Deviations from Spec
- None significant — design was followed exactly

### Pain Points
- TensorAccessorArgs placeholder size mismatch (1 vs 2 elements for interleaved) — not documented
- Packed bf16 vs raw float32 confusion for generate_reduce_scaler — the helper expects float32 bits, not packed bf16
- conftest.py device fixture needed keyword arg — inconsistent API

---

## Infrastructure Issues

- **No device hangs**: All tests ran cleanly with no hangs
- **No build issues**: Kernels compile at runtime, no build_metal.sh needed
- **No venv issues**: Python environment worked correctly
- **Git commit hooks**: Some analyzer agent commits failed (non-blocking) due to concurrent git operations

---

## Suggestions for Improving the Agent Pipeline

1. **TensorAccessorArgs placeholder docs**: Document that interleaved tensors produce 2 CT args, so absent tensor placeholders need `[0, 0]` not `[0]`
2. **Scaler format clarification**: The `generate_reduce_scaler` helper expects float32 bit patterns (via union), not packed bf16. This should be explicitly documented in the pipeline skill
3. **Conftest device fixture**: Should be auto-generated by the builder with the correct `device_id=` keyword argument
4. **Parallel analyzer commits**: Analyzers running in parallel can conflict on git commits — should use separate worktrees or serialize commits
5. **Stage count optimization**: 4 stages was optimal — the 6-stage plan from initial design was correctly consolidated by the architect
