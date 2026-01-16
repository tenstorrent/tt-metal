# ttnn-kernel-writer Execution Log

## Operation
layernorm_fused_rm

## Date
2026-01-16

## Status
PARTIAL

## Summary
Implemented reader, compute (simplified), and verified writer kernels for layernorm_fused_rm. Kernels compile successfully but hang during execution due to CB synchronization issue with tilize/pack_untilize flow.

## Design Document Followed
`/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/normalization/layernorm_fused_rm/kernel_design.md`

## Kernels Implemented

### Reader Kernel
**File**: `device/kernels/dataflow/reader_layernorm_fused_rm.cpp`

Implements:
- Scaler tile generation (1/W value packed as bfloat16)
- Epsilon tile generation (epsilon value packed as bfloat16)
- Gamma RM stick read (1D tensor)
- Beta RM stick read (1D tensor)
- Per-row input reading (32 sticks per tile row)

Uses TensorAccessor with `next_compile_time_args_offset()` chaining.

### Compute Kernel
**File**: `device/kernels/compute/layernorm_fused_rm.cpp`

Current implementation (simplified for debugging):
- Tilize input using `tilize_init/tilize_block/tilize_uninit`
- Pack untilize output using `pack_untilize_init/pack_untilize_block/pack_untilize_uninit`

Original full implementation (reverted due to hang):
- Tilize input
- Reduce mean (reduce<SUM, REDUCE_ROW>)
- Center values (sub_bcast_cols)
- Square centered values (mul_tiles self-multiply)
- Reduce variance (reduce<SUM, REDUCE_ROW>)
- Add epsilon + rsqrt (add_bcast_scalar + rsqrt_tile)
- Normalize (mul_bcast_cols)
- Pack untilize output

### Writer Kernel
**File**: `device/kernels/dataflow/writer_layernorm_fused_rm.cpp`

Verified from factory builder stub:
- Waits for 32 pages from cb_out_rm
- Writes each stick to DRAM via TensorAccessor
- Proper NOC barrier and CB pop

## CB Configuration (from Program Factory)
- c_0: Input RM sticks (32 * stick_size)
- c_1: Tiled input (2 * Wt * tile_size, double-buffered)
- c_2: Scaler (2 * tile_size)
- c_3: Epsilon (tile_size)
- c_4: Gamma RM stick (stick_size)
- c_5: Beta RM stick (stick_size)
- c_6: Tiled gamma (Wt * tile_size, unused)
- c_7: Tiled beta (Wt * tile_size, unused)
- c_16: Output RM sticks (32 * stick_size)
- c_24: Centered (Wt * tile_size)
- c_25: Mean (tile_size)
- c_26: Variance (tile_size)
- c_27: Inverse std (tile_size)

## Design Compliance Summary

| Phase | Directive | Implementation | Compliant |
|-------|-----------|----------------|-----------|
| Reader | NO HELPER | Raw CB ops for data movement | Yes |
| Tilize | USE HELPER | tilize_init/block/uninit | Yes |
| Reduce | USE HELPER | reduce_init/tile/uninit | N/A (not reached) |
| Binary ops | USE HELPER | sub_bcast_cols, mul_bcast_cols | N/A (not reached) |
| Pack untilize | USE HELPER | pack_untilize_init/block/uninit | Yes |
| Writer | NO HELPER | Raw NOC writes with TensorAccessor | Yes |

## Stage 7 Tests
**File**: `test_dev/test_stage7_kernel_correctness.py`

Test created but BLOCKED due to kernel hang:
- `test_identity_passthrough`: Tests tilize->pack_untilize passthrough

## Issues Encountered

### BLOCKING: Kernel Execution Hang
**Symptom**: Test times out (60s+) during operation execution
**Location**: After kernel compilation, during execution
**Suspected cause**: CB synchronization mismatch between tilize and pack_untilize

Investigation attempted:
1. Verified CB page sizes match (stick_size for RM CBs, tile_size for tiled CBs)
2. Verified cb_push/cb_pop counts match between producer/consumer
3. Tried fixed template parameters for pack_untilize (Wt=1)
4. Simplified compute to just tilize->pack_untilize passthrough

**Next debugging steps needed**:
1. Enable watcher to identify which kernel hangs (reader/compute/writer)
2. Add DPRINT to trace CB operations
3. Verify tilize_block and pack_untilize_block output data layout expectations

### Discovered: Wrong untilize function
Initially used `untilize_block()` which outputs tiled format, not row-major.
Fixed to use `pack_untilize_block()` which outputs row-major format.

## Commits
- `e1d0a7a281` - [ttnn-kernel-writer] stage 7: WIP layernorm_fused_rm kernels

## Deviations from Design

1. **Gamma/beta tilization skipped**: Design specified tilizing gamma/beta once at start. Current implementation pops them without processing due to 1D tensor tilization complexity and focus on basic passthrough debugging.

2. **Full layernorm computation not active**: Reverted to simple tilize->pack_untilize passthrough to isolate hang issue. Full computation code was implemented but needed to be removed for debugging.

## Recommendations for Next Steps

1. Use ttnn-riscv-debugger agent to identify hang location via watcher
2. Verify pack_untilize output CB page size requirements
3. Check if pack_untilize expects specific CB configuration (e.g., untilize-specific page organization)
4. Once passthrough works, incrementally add layernorm phases back
