# Self-Reflection: layer_norm_rm

## Metadata
| Field | Value |
|-------|-------|
| Operation | `layer_norm_rm` |
| Operation Path | `ttnn/ttnn/operations/layer_norm_rm` |
| Pipeline Phases Executed | 0 (Discovery), 1 (Analysis), 2 (Design), 3 (Build), 4 (TDD Kernels), 5 (Report) |
| Agents Invoked | orchestrator, ttnn-operation-analyzer (x3), ttnn-operation-architect, ttnn-generic-op-builder, ttnn-kernel-writer-tdd |
| Total Git Commits | 9 (this run: `a1f1bf9c31` through `79b825e1aa`) |
| Total Pipeline Duration | ~45 minutes (10:21 to 11:06 UTC) |
| Overall Result | SUCCESS |

**Note**: This is the latest run on the `mstaletovic/10_02_LN_TDD_BREAD` branch. Git history shows at least 6 prior runs of this operation dating back to 2026-02-10, indicating heavy iteration on both the pipeline and the operation itself. This analysis focuses on the most recent run (March 10, second run of the day).

---

## 1. Pipeline Execution Summary

### Phase Timeline

| Phase | Agent(s) | Duration | Status | Key Observations |
|-------|----------|----------|--------|------------------|
| 0: Discovery | orchestrator | ~3m | PASS | Selected 3 references: tilize (input_stage), untilize (output_stage), batch_norm (compute_core). Replaced softmax with batch_norm as compute reference -- good choice, closer semantic match. |
| 1: Analysis | ttnn-operation-analyzer (x3) | ~5m | PASS | 3 parallel analyses completed. Tilize analysis completed first (10:23:48 action), batch_norm took longest (completed 10:25:51). |
| 2: Design | ttnn-operation-architect | ~5m | PASS | Produced 388-line op_design.md with 10-phase compute pipeline, 13 CBs, 3 TDD stages. Two design decisions logged: cb_centered persistence strategy and SUM+1/W scaler. |
| 3: Build | ttnn-generic-op-builder | ~11m | PASS | Created 10 files. 1 free retry on kernel compilation (tensor_accessor.hpp include). 6/6 integration tests passed. |
| 4: TDD Kernels | ttnn-kernel-writer-tdd | ~21m | PASS | 3/3 stages passed. 3 hard attempts consumed (1 numerical mismatch, 1 runtime error, 1 hang). |
| 5: Report | orchestrator | ~2m | PASS | Generated REPORT.md |

### Agent Duration Breakdown

Duration calculation method: Primary source is breadcrumb `"event":"start"` and `"event":"complete"` timestamps. For agents without complete events, git commit timestamps are used as bounds.

| Agent | Start Time | End Time | Wall Duration | Retries | Active vs Debugging |
|-------|------------|----------|---------------|---------|---------------------|
| ttnn-operation-analyzer (untilize) | 10:21:11 | 10:24:12 | ~3m | 0 | 3m active |
| ttnn-operation-analyzer (batch_norm) | 10:21:51 | 10:25:51 | ~4m | 0 | 4m active |
| ttnn-operation-analyzer (tilize) | 10:21:55 | 10:25:55 | ~4m | 0 | 4m active |
| ttnn-operation-architect | 10:27:35 | 10:32:11 | ~5m | 0 | 5m active |
| ttnn-generic-op-builder | 10:34:16 | 10:43:41 | ~9m | 1 free | ~7m active, ~2m fixing tensor_accessor include |
| ttnn-kernel-writer-tdd | 10:46:44 | 11:04:36 | ~18m | 3 hard | ~10m productive coding, ~8m debugging |

### Duration Visualization

```
Phase 0  |##|                                             (~3m)
Phase 1  |########|                                       (~5m) 3 analyzers in parallel
Phase 2       |########|                                  (~5m)
Phase 3            |################|                     (~11m) includes 1 free retry
Phase 4                           |################################| (~21m)
Phase 5                                                         |##| (~2m)
         0    5    10   15   20   25   30   35   40   45 min

Longest phase: Phase 4 (21m) -- kernel implementation with 3 TDD stages and 3 hard debugging cycles
```

### Time Distribution

| Category | Duration | % of Total | Notes |
|----------|----------|------------|-------|
| Analysis (Phase 0-1) | ~8m | 18% | 3 parallel analyzers |
| Design (Phase 2) | ~5m | 11% | Single architect |
| Build (Phase 3) | ~11m | 24% | Includes 1 free retry |
| Kernel implementation (Phase 4) | ~21m | 47% | 3 TDD stages |
| -- Productive coding | ~10m | 22% | Writing kernel code that passed |
| -- Debugging/retries | ~8m | 18% | Hypothesis-fix-retest cycles |
| -- Test execution | ~3m | 7% | Running tests on device |
| Reporting (Phase 5) | ~2m | 4% | |
| **Total** | **~45m** | **100%** | |

---

## 2. What Went Well

### 1. Stage 0 (mean_subtract) passed on first attempt with zero retries

**Phase/Agent**: Phase 4, ttnn-kernel-writer-tdd
**Evidence**: `.tdd_state.json` shows `"attempts": 0` (meaning zero failures before passing) and empty `failure_history` for the mean_subtract stage. Breadcrumb timeline shows implementation started at 10:47:00 and test passed at 10:49:13 -- only ~2 minutes from first kernel code to passing test.
**Why it worked**: The design doc's Phase 1-3 specification was precise: CB indices, helper signatures, input/output policies, and manual pop requirements were all correctly specified. The kernel writer was able to translate the design directly into code without ambiguity.

### 2. All 13 CBs were correctly configured by the builder with zero CB-sizing bugs

**Phase/Agent**: Phase 3, ttnn-generic-op-builder
**Evidence**: The builder's execution log Section 2a shows all 13 CBs match the architect's design exactly (IDs, page sizes, page counts). None of the 3 TDD stage failures were caused by CB misconfiguration. The only builder issue was the spurious `tensor_accessor.hpp` include (a free retry, not CB-related).
**Why it worked**: The architect produced an explicit CB table with IDs, page counts, purposes, and lifetimes. The builder had an unambiguous specification to implement.

### 3. Hypothesis quality was consistently HIGH confidence and correct

**Phase/Agent**: Phase 4, ttnn-kernel-writer-tdd
**Evidence**: All 4 hypotheses (H1-H4 in the breadcrumbs) were tagged `"confidence":"HIGH"` and all were correct on first formulation. No LOW-confidence hypotheses, no multi-hypothesis investigation cycles, no wrong fixes that needed reverting. Specific examples:
- H2 (SCALAR vs COL broadcast): correctly identified from "systematic errors with different magnitudes per row"
- H4 (Bulk vs PerTile deadlock): correctly identified from "works for Wt=1 but hangs for Wt=4"
**Why it worked**: The kernel writer has good diagnostic reasoning. The error signals were informative (numerical patterns, shape-dependent behavior) rather than opaque.

### 4. Reference operation selection improved (batch_norm replaced softmax)

**Phase/Agent**: Phase 0, orchestrator
**Evidence**: Prior runs (Feb 10, Mar 3, Mar 5, Mar 6) used `softmax` as the compute_core reference. This run used `batch_norm`, which is a closer semantic match -- both are normalization operations that apply per-channel/per-row statistics. The batch_norm analysis (33KB document) provided directly applicable patterns: CB lifetime tiers, binary_dest_reuse_tiles, dynamic routing for optional affine params.
**Why it worked**: The discovery phase was improved between runs to select more semantically relevant references.

### 5. Clean inter-agent handoffs with minimal friction

**Phase/Agent**: All handoffs
**Evidence**: The builder's execution log under "Interpretation Issues" says "None - input was clear and complete." The kernel writer's breadcrumbs show only 2 design doc deviations (SCALAR->COL broadcast, Bulk->PerTile output), both of which were design errors rather than ambiguities. No `upstream_feedback` events from the kernel writer complaining about builder output quality.
**Why it worked**: The architect's design doc was structured with explicit tables for CB allocation, kernel arguments, helper signatures, and binary op broadcast verification.

---

## 3. Issues Found

### Issue 1: Design doc specified SCALAR broadcast for inv_std multiplication, but REDUCE_ROW output requires COL broadcast

| Field | Value |
|-------|-------|
| Severity | HIGH |
| Phase / TDD Stage | Phase 4 -- full_normalize (stage 1) |
| Agent | ttnn-operation-architect (root cause), ttnn-kernel-writer-tdd (detected and fixed) |
| Retries Consumed | 1 hard attempt |
| Time Cost | ~3 minutes (10:51:51 test fail to 10:54:05 test pass) |

**Problem**: The design doc `op_design.md` line 183 specified `Mul inv_std | MUL | All (cb_centered) | Scalar (cb_inv_std) | SCALAR` in the Binary Op Broadcast Verification table. Phase 7 pseudocode on line 316-317 also showed `BroadcastDim::SCALAR`. However, `cb_inv_std` is produced by Phase 6 (add_eps+rsqrt) which operates on `cb_var`, a REDUCE_ROW output. REDUCE_ROW stores per-row values in column 0 of the tile, not at scalar position [0,0]. Using SCALAR broadcast replicates only the row-0 value across all positions, producing wrong normalization for rows 1-31.

The kernel writer's hypothesis H2 (breadcrumb at 10:53:18) correctly diagnosed this: "SCALAR bcast uses only [0][0]=rsqrt(var_row0+eps) for ALL rows. Need COL broadcast to apply each row's rsqrt(var+eps) correctly."

**Root Cause**: The architect confused tile data placement semantics. REDUCE_ROW output has valid data in column 0 (all rows), making it COL-broadcastable. SCALAR broadcast only uses the [0][0] element. The design doc has a "Binary Op Broadcast Verification" table that should have caught this but incorrectly labeled the CB as "Scalar (cb_inv_std)" when the actual valid region is col0 (inherited from cb_var's REDUCE_ROW output).

**Fix for agents**:
- **ttnn-operation-architect**: Add a validation rule: "When a CB is populated by REDUCE_ROW output, its valid region is Col0, and any downstream binary op must use COL broadcast (not SCALAR) to distribute the per-row values." This rule should be explicitly checked during the Binary Op Broadcast Verification table construction. This is the same pattern as cb_mean (Phase 3 correctly uses COL broadcast), so the architect was inconsistent within the same design.

### Issue 2: Bulk output policy for in-place CB reuse deadlocks when Wt > 1

| Field | Value |
|-------|-------|
| Severity | HIGH |
| Phase / TDD Stage | Phase 4 -- affine_transform (stage 2) |
| Agent | ttnn-operation-architect (root cause), ttnn-kernel-writer-tdd (detected and fixed) |
| Retries Consumed | 1 hard attempt (hang, required device reset) |
| Time Cost | ~3 minutes (11:00:21 hang detected to 11:01:09 fix applied) + device reset time |

**Problem**: Design doc lines 338-340 (Phase 8) and 358-359 (Phase 9) specified `BinaryOutputPolicy::Bulk` for gamma/beta in-place operations on `cb_out_pre_untilize`. The Bulk policy reserves all Wt output pages upfront before any computation. When input A is the same CB as output (in-place), the CB already has Wt input tiles occupying all pages. Bulk cannot reserve Wt more pages because the CB total capacity is Wt. This works for Wt=1 (pop frees the single input tile before the reserve blocks) but deadlocks for Wt>1.

The hang manifested on shape (1,1,64,128) where Wt=4. Shape (1,1,32,32) with Wt=1 passed, masking the bug.

**Root Cause**: The architect did not model in-place CB capacity constraints when selecting output policies. The design doc note on line 346 ("In-place CB reuse requires that the helper consumes A tiles before pushing output. WaitAndPopPerTile for A ensures this.") is correct about input policy but failed to apply the same reasoning to output policy. The architect even noted that "WaitAndPopPerTile on A ensures tiles are consumed before output tiles occupy the same slots" but did not realize Bulk output pre-reserves all output slots before any A tiles are consumed.

**Fix for agents**:
- **ttnn-operation-architect**: Add a hard rule: "When input CB == output CB (in-place reuse), ALWAYS use BinaryOutputPolicy::PerTile (never Bulk). Bulk reserves all output pages upfront, which deadlocks when the CB is already full with input tiles."
- **ttnn-operation-architect**: During design review, flag any phase where the same CB appears as both input and output in a binary op and validate the output policy is PerTile.

### Issue 3: ttnn.tilize() fails on small tensors with volume < TILE_HW

| Field | Value |
|-------|-------|
| Severity | MEDIUM |
| Phase / TDD Stage | Phase 4 -- affine_transform (stage 2) |
| Agent | ttnn-generic-op-builder (should have handled) and ttnn-kernel-writer-tdd (fixed) |
| Retries Consumed | 1 hard attempt |
| Time Cost | ~1 minute (quick diagnosis from clear error message) |

**Problem**: The entry point `layer_norm_rm.py` called `ttnn.tilize()` on gamma/beta tensors of shape (1,1,1,W). For the minimal test shape (W=32), volume is 32, which is less than TILE_HW (1024). `ttnn.tilize()` requires `physical_volume % TILE_HW == 0`. The fix was to pad H from 1 to 32 using `ttnn.pad` before tilizing.

Error message: `TT_FATAL @ tilize_device_operation.cpp:31: input_tensor_a.physical_volume() % tt::constants::TILE_HW == 0`

**Root Cause**: Neither the architect nor the builder anticipated that gamma/beta tensors (shape 1,1,1,W) would need padding before tilize. The architect's design doc (line 217) mentioned gamma/beta tile reads but did not specify how RM gamma/beta tensors get converted to tiles. The builder's handoff notes (execution log line 205) even flagged "gamma/beta are RM tensors (not TILE_LAYOUT). The reader needs to handle them as RM data when loading into CB25/CB26" -- but this was wrong advice (the kernel writer correctly decided to tilize them on the host side, which requires padding).

**Fix for agents**:
- **ttnn-operation-architect**: When specifying optional parameter tensors (gamma, beta, weight, bias) that need to be read as tiles, explicitly specify the host-side preprocessing: "Pad to tile-aligned shape if necessary, then tilize before passing to program descriptor."
- **ttnn-generic-op-builder**: When the design doc mentions optional tensor inputs smaller than tile size, add padding+tilize logic in the entry point as part of infrastructure setup.

### Issue 4: tensor_accessor.hpp include path does not exist for device kernels

| Field | Value |
|-------|-------|
| Severity | LOW |
| Phase / TDD Stage | Phase 3 -- Build |
| Agent | ttnn-generic-op-builder |
| Retries Consumed | 1 free retry |
| Time Cost | ~2 minutes |

**Problem**: The builder included `ttnn/cpp/ttnn/tensor/accessor/tensor_accessor.hpp` in the reader and writer stub kernels. This header does not exist in the device kernel include path. TensorAccessor types are already provided by `api/dataflow/dataflow_api.h`.

**Root Cause**: The builder's agent instructions contain an incorrect include mapping table. The builder's own execution log (Recommendation 1) flagged this and suggested removing the mapping.

**Fix for agents**:
- **ttnn-generic-op-builder**: Remove the `tensor_accessor.hpp` include from the instruction's include mapping table. Add a note: "TensorAccessor is automatically available via api/dataflow/dataflow_api.h -- no explicit include needed."

---

## 4. Efficiency Analysis

### Per-TDD-Stage Breakdown

| TDD Stage | Duration | Attempts (free/hard) | Result | Bottleneck |
|-----------|----------|---------------------|--------|------------|
| mean_subtract | ~3m (10:47:00-10:49:19) | 0 free, 0 hard | PASS | Clean -- first-attempt pass |
| full_normalize | ~5m (10:50:15-10:54:12) | 0 free, 1 hard | PASS | Numerical mismatch from design error (SCALAR vs COL broadcast) |
| affine_transform | ~10m (10:54:49-11:04:36) | 0 free, 2 hard | PASS | Runtime error + hang from upstream python fixes and design error |

### Time Sinks

| Rank | Area | Agent | Duration | % of Total | Description | Retry Count | Likely Cause |
|------|------|-------|----------|------------|-------------|-------------|--------------|
| 1 | affine_transform stage | kernel-writer-tdd | ~10m | 22% | 3 issues in sequence: upstream Python fixes (tilize padding, TensorAccessorArgs), runtime error, hang | 2 hard | Design doc omitted gamma/beta preprocessing; Bulk output policy bug |
| 2 | Build phase | generic-op-builder | ~9m | 20% | Creating 10 files + 1 free retry on tensor_accessor include | 1 free | Known agent instruction bug |
| 3 | full_normalize stage | kernel-writer-tdd | ~5m | 11% | SCALAR vs COL broadcast debugging | 1 hard | Design doc error in broadcast table |

### Wasted Work

| Agent | What Was Done | Why It Was Wasted | How to Avoid |
|-------|--------------|-------------------|--------------|
| kernel-writer-tdd | Implemented Phase 7 with SCALAR broadcast per design doc | Had to change to COL broadcast after numerical mismatch | Architect should validate REDUCE_ROW output -> COL broadcast rule |
| kernel-writer-tdd | Implemented Phases 8-9 with Bulk output per design doc | Had to change to PerTile output after hang | Architect should flag in-place CB reuse -> PerTile output rule |
| generic-op-builder | Added tensor_accessor.hpp include in stub kernels | Had to remove it after compilation failure | Fix agent instructions |
| kernel-writer-tdd | Builder's handoff said gamma/beta should be RM in reader | Kernel writer tilized them on host instead (correct approach) | Architect should specify gamma/beta preprocessing explicitly |

---

## 5. Inter-Agent Communication Issues

### Handoff 1: orchestrator -> ttnn-operation-analyzer (x3)

| Field | Value |
|-------|-------|
| Artifact Passed | Reference operation file paths |
| Quality | GOOD |
| Issues | None observed. All 3 analyzers completed successfully with relevant analyses. |
| Downstream Impact | None negative. Batch_norm analysis was especially useful for the architect. |
| Suggestion | None needed. |

### Handoff 2: ttnn-operation-analyzer -> ttnn-operation-architect

| Field | Value |
|-------|-------|
| Artifact Passed | tilize_analysis.md, untilize_analysis.md, batch_norm_analysis.md |
| Quality | GOOD |
| Issues | None. Architect breadcrumbs show mode was correctly detected as "Hybrid" with all 3 references recognized. |
| Downstream Impact | Architect produced a comprehensive design. The batch_norm analysis directly informed CB lifetime management and optional affine parameter handling. |
| Suggestion | None needed. |

### Handoff 3: ttnn-operation-architect -> ttnn-generic-op-builder

| Field | Value |
|-------|-------|
| Artifact Passed | op_design.md (Part 1: Architecture) |
| Quality | GOOD |
| Issues | Minor: The design doc does not specify how gamma/beta RM tensors should be converted to TILE_LAYOUT for CB reads. The builder interpreted this as "pass RM tensors through; reader handles RM data" which was incorrect. |
| Downstream Impact | The builder's handoff note to the kernel writer said "gamma/beta are RM tensors (not TILE_LAYOUT). The reader needs to handle them as RM data." This was wrong -- the kernel writer had to tilize them on the host side, requiring upstream Python modifications. |
| Suggestion | Architect should explicitly specify the gamma/beta preprocessing pipeline: pad to tile-aligned shape, tilize, pass TILE_LAYOUT tensors to program descriptor. |

### Handoff 4: ttnn-generic-op-builder -> ttnn-kernel-writer-tdd

| Field | Value |
|-------|-------|
| Artifact Passed | Stub kernels, program_descriptor.py, layer_norm_rm.py, test files |
| Quality | ADEQUATE |
| Issues | (1) Missing gamma/beta TensorAccessorArgs in reader compile-time args -- kernel writer had to add them. (2) Wrong advice about gamma/beta being RM in the reader. (3) tensor_accessor.hpp include needed removal (already fixed before handoff). |
| Downstream Impact | Kernel writer spent time on upstream Python fixes (3 `upstream_fix` breadcrumb events in affine_transform stage) instead of purely writing kernel code. |
| Suggestion | Builder should include TensorAccessorArgs for all tensor inputs mentioned in the design doc, not just input/output. The builder should follow the design doc's kernel argument table, which lists gamma/beta addresses. |

### Handoff 5: ttnn-operation-architect -> ttnn-kernel-writer-tdd

| Field | Value |
|-------|-------|
| Artifact Passed | op_design.md (Part 2: Kernel Implementation) |
| Quality | ADEQUATE -- two design errors found |
| Issues | (1) Phase 7 SCALAR broadcast should be COL (line 183, 316-317). (2) Phases 8-9 Bulk output should be PerTile (lines 338-340, 358-359). The design doc's Binary Op Broadcast Verification table (line 183) was internally inconsistent -- it correctly specified COL for Phase 3 (subtract mean) but incorrectly specified SCALAR for Phase 7 (mul inv_std), despite both CBs being REDUCE_ROW outputs with the same col0 valid region. |
| Downstream Impact | 2 hard attempts consumed. ~6 minutes of debugging. One device hang requiring reset. |
| Suggestion | Add systematic validation: if a CB is produced by REDUCE_ROW, tag its valid region as "col0" and propagate that through all downstream broadcast decisions. Flag any mismatch between valid region and broadcast dim. |

---

## 6. Upstream Feedback Synthesis

### Agent Instruction Improvements

| Target | Source Agent | Recommendation | Confidence | Priority |
|--------|-------------|----------------|------------|----------|
| ttnn-generic-op-builder | ttnn-generic-op-builder | Remove tensor_accessor.hpp from include mapping table; TensorAccessor available via dataflow_api.h | HIGH | MEDIUM |
| ttnn-generic-op-builder | ttnn-generic-op-builder | Recommend `api/compute/common.h` as standard compute kernel include instead of compute_kernel_hw_startup.h | HIGH | LOW |
| ttnn-operation-architect | ttnn-kernel-writer-tdd | REDUCE_ROW output CB valid region is col0 -> must use COL broadcast (not SCALAR) for downstream binary ops | HIGH | HIGH |
| ttnn-operation-architect | ttnn-kernel-writer-tdd | In-place CB reuse (input == output CB) requires PerTile output policy, never Bulk | HIGH | HIGH |
| ttnn-operation-architect | ttnn-kernel-writer-tdd | Explicitly specify preprocessing for optional tensor params (gamma/beta): pad + tilize if needed | HIGH | MEDIUM |

### Pipeline-Level Improvements

| Area | Observation | Recommendation | Priority |
|------|-------------|----------------|----------|
| Design validation | Two design errors consumed 2 hard attempts and caused a device hang | Add automated broadcast-dimension validation: cross-check CB producer (reduce dim) against consumer (broadcast dim) | HIGH |
| Build completeness | Builder missed TensorAccessorArgs for gamma/beta | Builder should include TensorAccessorArgs for ALL tensor inputs in the design doc, not just primary input/output | MEDIUM |
| TDD stage ordering | The affine_transform stage had 3 different failure types layered together (runtime error, hang, upstream fix) | Consider splitting affine into two sub-stages: "gamma_only" and "gamma_beta" to isolate issues | LOW |
| Logging | No execution log from kernel-writer-tdd (only breadcrumbs) | Kernel writer should produce a structured execution log like the builder does | MEDIUM |

---

## 7. Comparison with Known Issues

### Known Issues Encountered

| Issue # | Title | Encountered? | Notes |
|---------|-------|-------------|-------|
| 1 | Kernel writer burns massive context on numerical debugging | PARTIALLY | One numerical mismatch in full_normalize, but diagnosed quickly (~3m). HIGH-confidence hypothesis on first attempt. This run did NOT exhibit the long-grinding behavior described in the issue. |
| 2 | Too many planning stages (long leash) | NO (DONE) | Architect merge is working well -- single agent produced complete design in ~5m. |
| 3 | .tdd_state.json coupling fragility | NO | No schema issues observed. |
| 4 | No fast path for simple operations | NO | Layer_norm_rm is medium complexity; fast path would not apply. |
| 6 | Builder runs on Sonnet | POSSIBLY | The builder's tensor_accessor.hpp include error is consistent with the kind of detail-sensitivity issue described. However, it was a free retry and quickly resolved. |
| 7 | Discovery keyword matching | NO | Discovery correctly selected batch_norm over softmax this run. |
| 9 | No validation between architect and builder output | YES | The builder correctly implemented all 13 CBs, but missed gamma/beta TensorAccessorArgs. A static cross-check between op_design.md kernel argument tables and program_descriptor.py compile-time args would have caught this. |
| 11 | No incremental re-run capability | NO | Pipeline completed successfully; no re-run needed. |

### New Issues Discovered

| Title | Description | Suggested Priority |
|-------|-------------|-------------------|
| REDUCE_ROW output -> broadcast dim validation gap | Architect specified SCALAR broadcast for a REDUCE_ROW-produced CB. This is a semantic error: REDUCE_ROW outputs have per-row values in col0, requiring COL broadcast. The architect has no rule enforcing this consistency. | HIGH |
| In-place CB reuse with Bulk output policy deadlocks for Wt > 1 | Design doc specified Bulk output for in-place CB operations (Phases 8-9). Bulk reserves all output pages upfront, deadlocking when the CB is already full with input tiles. Only PerTile is safe for in-place reuse. | HIGH |
| Optional tensor preprocessing not specified in design | Architect did not specify how gamma/beta (1,1,1,W) RM tensors should be converted to tile-readable format. This left the builder and kernel writer to figure it out independently, leading to conflicting approaches (builder: leave as RM, kernel writer: pad+tilize on host). | MEDIUM |

---

## 8. Actionable Recommendations

### Recommendation 1: Add REDUCE_ROW -> COL broadcast validation rule to architect

- **Type**: instruction_change
- **Target**: ttnn-operation-architect agent prompt / design validation checklist
- **Change**: Add rule: "When populating the Binary Op Broadcast Verification table, check: if a CB was produced by a REDUCE_ROW operation, its valid region is Col0 and any downstream binary op consuming it must use BroadcastDim::COL (not SCALAR). SCALAR only replicates [0][0]; COL replicates each row's col0 value across all columns."
- **Expected Benefit**: Prevents the most common numerical mismatch error observed across multiple runs. This same error appeared in this run's full_normalize stage.
- **Priority**: HIGH
- **Effort**: SMALL

### Recommendation 2: Add in-place CB reuse -> PerTile output rule to architect

- **Type**: instruction_change
- **Target**: ttnn-operation-architect agent prompt / CB policy decision rules
- **Change**: Add rule: "When input CB == output CB for a binary op (in-place reuse), ALWAYS specify BinaryOutputPolicy::PerTile. BinaryOutputPolicy::Bulk reserves all output pages upfront, which deadlocks when Wt > 1 because the CB is already full with input pages. Only PerTile is safe for in-place reuse."
- **Expected Benefit**: Prevents device hangs on in-place CB operations. This pattern appears whenever optional affine transforms are applied in-place.
- **Priority**: HIGH
- **Effort**: SMALL

### Recommendation 3: Require explicit preprocessing spec for optional tensor params

- **Type**: instruction_change
- **Target**: ttnn-operation-architect agent prompt
- **Change**: Add a "Parameter Preprocessing" section to the design template. For each optional tensor parameter (gamma, beta, weight, bias), specify: (1) input format (RM/TILE), (2) host-side preprocessing needed (pad dimensions, tilize, etc.), (3) resulting format/shape when passed to program descriptor. Example: "gamma (1,1,1,W) RM -> pad H to 32 -> tilize -> (1,1,32,W) TILE passed to reader."
- **Expected Benefit**: Eliminates the handoff confusion where the builder thinks gamma should be RM but the kernel writer knows it needs to be tiled.
- **Priority**: MEDIUM
- **Effort**: SMALL

### Recommendation 4: Add TensorAccessorArgs completeness check to builder

- **Type**: new_validation
- **Target**: ttnn-generic-op-builder agent prompt
- **Change**: After creating compile-time args, cross-check against the design doc's kernel argument table. Every tensor address passed as a runtime arg to a dataflow kernel should have a corresponding TensorAccessorArgs in the compile-time args. Flag any missing ones.
- **Expected Benefit**: Prevents the kernel writer from needing to make upstream Python fixes during TDD, which disrupts the flow and adds debugging time.
- **Priority**: MEDIUM
- **Effort**: SMALL

### Recommendation 5: Remove tensor_accessor.hpp from builder's include mapping

- **Type**: instruction_change
- **Target**: ttnn-generic-op-builder agent prompt, include mapping table
- **Change**: Remove the entry mapping TensorAccessor to `ttnn/cpp/ttnn/tensor/accessor/tensor_accessor.hpp`. Replace with note: "TensorAccessor is automatically available via api/dataflow/dataflow_api.h -- no explicit include needed for dataflow kernels."
- **Expected Benefit**: Eliminates 1 free retry on every operation build that uses TensorAccessor.
- **Priority**: MEDIUM
- **Effort**: SMALL

### Recommendation 6: Kernel writer should produce structured execution log

- **Type**: pipeline_change
- **Target**: ttnn-kernel-writer-tdd agent prompt
- **Change**: Require the kernel writer to produce a `ttnn-kernel-writer-tdd_execution_log.md` with the same structure as the builder's execution log: input interpretation, per-stage timeline, recovery table, deviations, handoff notes. Currently only breadcrumbs are produced.
- **Expected Benefit**: Better observability for self-reflection analysis. The builder's execution log was the most informative artifact for analyzing Phase 3; the kernel writer lacks an equivalent.
- **Priority**: MEDIUM
- **Effort**: MEDIUM

---

## 9. Overall Assessment

### Pipeline Maturity Score

| Dimension | Score | Notes |
|-----------|-------|-------|
| Discovery accuracy | 4/5 | Selected batch_norm over softmax -- better semantic match. Could improve by using LLM reasoning instead of keyword matching. |
| Analysis quality | 4/5 | All 3 analyses were thorough (15-33KB each). Batch_norm analysis correctly highlighted key patterns reused by the architect. |
| Design completeness | 3/5 | Comprehensive structure (388 lines, 13 CBs, 10 phases) but contained 2 errors: wrong broadcast dim and wrong output policy. Both were caught by kernel writer but cost 2 hard attempts. |
| Build correctness | 4/5 | 13 CBs correct, 6/6 integration tests. Minor gaps: missing gamma/beta TensorAccessorArgs and wrong gamma/beta format advice. 1 free retry on known include bug. |
| Kernel implementation | 4/5 | 3/3 stages passed with only 3 hard attempts (16.7% of budget). All hypotheses were correct on first formulation. Upstream fixes were clean and well-documented. |
| Inter-agent communication | 3/5 | Handoffs 1-3 were good. Handoff 4 (builder->kernel writer) had gaps (missing TensorAccessorArgs, wrong gamma/beta advice). Handoff 5 (architect->kernel writer) had 2 design errors. |
| Logging/observability | 4/5 | Breadcrumbs for all agents with timestamps. Builder has full execution log. Kernel writer lacks execution log (only breadcrumbs). Analyzer breadcrumbs are minimal but sufficient. No gaps in timeline reconstruction. |

### Top 3 Things to Fix

1. **Add REDUCE_ROW -> COL broadcast validation to architect instructions.** This is a recurring error (appeared in this run and is a common misunderstanding). A simple rule eliminates a hard debugging cycle every time.

2. **Add in-place CB reuse -> PerTile output policy rule to architect instructions.** In-place operations with Bulk policy are a guaranteed hang for Wt > 1. A hard rule prevents device hangs and reset cycles.

3. **Require explicit parameter preprocessing specification in design doc.** The gamma/beta format confusion between architect, builder, and kernel writer caused unnecessary upstream fixes and conflicting handoff notes. Making preprocessing explicit in the design eliminates this class of inter-agent confusion.

### What Worked Best

The **TDD stage progression design** was the single strongest aspect of this pipeline run. The architect decomposed layer_norm_rm into 3 stages of increasing complexity (mean_subtract, full_normalize, affine_transform), each building on the previous one. Stage 0 passed on first attempt, validating the entire data pipeline (reader/compute/writer, tilize/untilize, reduce/broadcast). This meant that when stages 1 and 2 failed, the kernel writer could confidently isolate the bug to the newly added code rather than questioning the entire pipeline. The progressive TDD approach transformed what could have been an opaque "nothing works" debugging session into targeted, quick-to-resolve issues.
