# Execution Log: ttnn-kernel-designer

## 1. Metadata

| Field | Value |
|-------|-------|
| **Operation** | centralize_w_rm |
| **Agent** | ttnn-kernel-designer |
| **Status** | SUCCESS |
| **Input Files** | centralize_w_rm_spec.md |
| **Output Files** | kernel_design.md |
| **Predecessor** | ttnn-factory-builder |
| **Successor** | ttnn-kernel-writer |

## 2. Input Interpretation

| Field | Value | Confidence | Notes |
|-------|-------|------------|-------|
| operation_name | centralize_w_rm | HIGH | Explicitly stated in spec |
| operation_flow | tilize -> reduce -> bcast_sub -> untilize | HIGH | 4 phases in single compute kernel |
| cb_persistence | CB_1 must persist through reduce | HIGH | Critical for bcast_sub to access original data |
| broadcast_dim | COL | HIGH | REDUCE_ROW output is column-shaped |

### Upstream Feedback
None - spec was well-structured and accurate.

## 2a. Helper Library Analysis

| Helper File | Read? | Applicable? | Functions Identified |
|-------------|-------|-------------|---------------------|
| tilize_helpers.hpp | YES | YES | `tilize(icb, block_w, ocb, num_blocks)` |
| untilize_helpers.hpp | YES | YES | `untilize<tile_width, icb_id, ocb_id>(num_rows)` |
| reduce_helpers.hpp | YES | YES | `reduce<PoolType, ReduceDim, InputMode>(icb, icb_scaler, ocb, TileShape)` |
| binary_op_helpers.hpp | YES | YES | `sub<BroadcastDim, InputAPolicy, InputBPolicy>(icb_a, icb_b, ocb, shape)` |
| dest_helpers.hpp | YES | YES | `DEST_AUTO_LIMIT` for untilize dispatch |

### Phase-to-Helper Mapping

| Phase | Implementation Approach | Rationale |
|-------|------------------------|-----------|
| Tilize | USE HELPER: `tilize()` | Standard RM to TILE conversion |
| Reduce | USE HELPER: `reduce<SUM, REDUCE_ROW, PERSISTENT>()` | PERSISTENT mode keeps tiles for bcast_sub |
| BcastSub | USE HELPER: `sub<COL, Preloaded, Streaming>()` | COL broadcast for column-shaped mean |
| Untilize | USE HELPER: `untilize<Wt>()` | Standard TILE to RM conversion |

### Encapsulation Notes

For phases marked "USE HELPER", documented that helpers handle:
- [x] CB wait/pop/reserve/push
- [x] DST register management
- [x] Init/uninit sequences

## 3. Execution Timeline

### Attempt 1: Design Document Creation (SUCCESS)

| Step | Action | Result |
|------|--------|--------|
| 1 | Read spec file | Understood 4-phase operation |
| 2 | Read all helper headers | Identified applicable functions |
| 3 | Validate spec | Found 2 issues requiring resolution |
| 4 | Create Data Semantics Model | Documented CB contents and valid regions |
| 5 | Verify broadcast semantics | Confirmed BroadcastDim::COL is correct |
| 6 | Write kernel_design.md | SUCCESS |

## 4. Recovery Summary

No recovery needed - first attempt succeeded.

| Stage | Attempts | Final Result |
|-------|----------|--------------|
| Design | 1 | SUCCESS |

## 5. Spec Validation Issues Found

### Issue 1: Reduce Helper CB Persistence
- **Problem**: Default STREAMING mode would pop CB_1 tiles, destroying them before bcast_sub
- **Resolution**: Use `ReduceInputMode::PERSISTENT` to keep tiles in CB_1

### Issue 2: Binary Op Input Policy
- **Problem**: Default Streaming policy would wait/pop tiles that are already present
- **Resolution**: Use `cb_policies::Preloaded` for input A (CB_1) in sub() call

## 6. Deviations

None - followed system instructions exactly.

## 7. Artifacts

### Created Files
| File | Purpose |
|------|---------|
| `kernel_design.md` | Kernel design document for kernel-writer |

## 8. Key Design Decisions

### Decision 1: PERSISTENT Mode for Reduce
- **What**: Use `ReduceInputMode::PERSISTENT` instead of default `STREAMING`
- **Why**: CB_1 tiles must remain available for bcast_sub after reduce
- **Impact**: Reduce helper will NOT pop CB_1 tiles

### Decision 2: Preloaded Policy for BcastSub Input A
- **What**: Use `cb_policies::Preloaded` for CB_1 in sub() call
- **Why**: CB_1 tiles are already present from tilize and persistent from reduce
- **Impact**: Binary op helper uses indexed access, pops at end

### Decision 3: BroadcastDim::COL Verification
- **What**: Verified BroadcastDim::COL is correct for this operation
- **Why**: REDUCE_ROW produces column-shaped output (Col0 valid), must broadcast right
- **Impact**: Mean is correctly broadcast across all width positions

## 9. Handoff Notes for ttnn-kernel-writer

### Critical Implementation Notes

1. **CB_1 Persistence**: The reduce call MUST use `ReduceInputMode::PERSISTENT` or CB_1 tiles will be consumed before bcast_sub can use them.

2. **Binary Op Policies**: The sub() call needs:
   - Input A (CB_1): `cb_policies::Preloaded` - tiles already present, use indexed access
   - Input B (CB_3): `cb_policies::Streaming` - default for COL broadcast

3. **CB Push/Pop Verification**:
   - CB_1: tilize pushes Wt, reduce does NOT pop (PERSISTENT), bcast_sub pops Wt
   - CB_3: reduce pushes 1, bcast_sub pops 1
   - All other CBs: standard push/pop matching

4. **Reference Kernels**: `reduce_mean_w_rm` kernels are a good starting point, but this operation has additional complexity from:
   - The bcast_sub phase (not in reduce_mean_w_rm)
   - CB_1 persistence requirement
   - Full-width output (not reduced width)

### Template Parameters Summary
```cpp
// Phase 2: Reduce
compute_kernel_lib::reduce<
    PoolType::SUM,
    ReduceDim::REDUCE_ROW,
    ReduceInputMode::PERSISTENT>(...)

// Phase 3: BcastSub
compute_kernel_lib::sub<
    BroadcastDim::COL,
    cb_policies::Preloaded,
    cb_policies::Streaming>(...)
```

## 10. Instruction Recommendations

### What Worked Well
1. Having reduce_mean_w_rm as a reference operation
2. Clear spec with CB IDs and data flow
3. Kernel helper library documentation with @example blocks

### Suggestions for Improvement
1. **Spec Enhancement**: For operations with CB persistence requirements, explicitly state the required `ReduceInputMode` in the spec
2. **Helper Docs**: Document which input policies are compatible with which scenarios (e.g., Preloaded requires tiles already in CB)

## 11. Git Commit History

| Commit SHA | Message | Files |
|------------|---------|-------|
| (pending) | [ttnn-kernel-designer] design: centralize_w_rm | kernel_design.md |
