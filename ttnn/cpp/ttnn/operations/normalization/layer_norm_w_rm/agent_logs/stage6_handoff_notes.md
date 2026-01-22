# Stage 6 Handoff Notes for layer_norm_w_rm

## Completion Status
- **Stage 6: COMPLETE**
- **All tests passed**: 2/2
- **CB synchronization verified**: No deadlocks, all push/pop counts balanced
- **Kernels compile at runtime**: Successfully

## Files Created/Modified

### Kernel Files Created
1. `device/kernels/dataflow/reader_layer_norm_w_rm.cpp` - Reader kernel (STUB)
2. `device/kernels/compute/layer_norm_w_rm_compute.cpp` - Compute kernel (STUB)
3. `device/kernels/dataflow/writer_layer_norm_w_rm.cpp` - Writer kernel (STUB)

### Program Factory Modified
- `device/layer_norm_w_rm_program_factory.cpp` - Complete with kernel creation and runtime args

## Circular Buffer Configuration

### CB Summary (16 CBs total: c_0 to c_14, c_16)

| CB ID | Name | Purpose | Page Size | Num Pages | Producer | Consumer | Lifetime |
|-------|------|---------|-----------|-----------|----------|----------|----------|
| c_0 | cb_in_rm | Input RM sticks | tile_size | 2*Wt | Reader | Compute | Block |
| c_1 | cb_in_tiled | Tiled input | tile_size | Wt | Compute | Compute | PERSISTENT (Phases 1-3) |
| c_2 | cb_scaler | Scaler (1/W) | tile_size | 1 | Reader | Compute | Program |
| c_3 | cb_mean_tiled | Mean tile | tile_size | 1 | Compute | Compute | Block |
| c_4 | cb_centralized | Centralized tiles | tile_size | Wt | Compute | Compute | PERSISTENT (Phases 3-8) |
| c_5 | cb_squared | Squared tiles | tile_size | Wt | Compute | Compute | Block |
| c_6 | cb_variance | Variance tile | tile_size | 1 | Compute | Compute | Block |
| c_7 | cb_epsilon | Epsilon scalar | tile_size | 1 | Reader | Compute | Program |
| c_8 | cb_rsqrt | Rsqrt result | tile_size | 1 | Compute | Compute | Block |
| c_9 | cb_standardized / cb_output_tiled | Standardized / Final output | tile_size | Wt | Compute | Compute / Untilize | Block (reused) |
| c_10 | cb_gamma_rm | Gamma RM sticks | tile_size | Wt | Reader | Compute | Program |
| c_11 | cb_gamma_tiled | Gamma tiled | tile_size | Wt | Compute | Compute | Program (persist) |
| c_12 | cb_beta_rm | Beta RM sticks | tile_size | Wt | Reader | Compute | Program |
| c_13 | cb_beta_tiled | Beta tiled | tile_size | Wt | Compute | Compute | Program (persist) |
| c_14 | cb_scaled | Scaled output | tile_size | Wt | Compute | Compute | Block |
| c_16 | cb_out_rm | Output RM sticks | tile_size | 2*Wt | Compute | Writer | Block |

## Current Stub Behavior

### Reader Kernel (STUB)
- Reads input sticks: Ht tile-rows, each 32 sticks of width W
- Reads gamma sticks: ONCE at program start, 32 sticks of width W
- Reads beta sticks: ONCE at program start, 32 sticks of width W
- Generates scaler tile (1/W): ONCE at program start
- Generates epsilon tile: ONCE at program start
- Pushes: input to c_0 (Ht*Wt pages), gamma to c_10 (Wt pages), beta to c_12 (Wt pages), scaler to c_2 (1 page), epsilon to c_7 (1 page)

### Compute Kernel (STUB)
- Waits for gamma_rm (c_10, Wt pages) and beta_rm (c_12, Wt pages) at program start
- Per tile-row:
  - Waits for input (c_0, Wt pages)
  - Copies RM data to output (c_16, Wt pages) using copy_tile
  - Pops input, pushes output
- **Does NOT implement any computation** - just passthrough for CB sync verification

### Writer Kernel (STUB)
- Writes output sticks: Ht tile-rows, each 32 sticks of width W
- Waits for output from c_16 (Wt pages per tile-row)
- Writes to DRAM using TensorAccessor

## CB Synchronization Verification

All CB push/pop counts are balanced:
- **c_0**: Reader pushes Ht*Wt, Compute pops Ht*Wt ✓
- **c_2**: Reader pushes 1, never popped (program lifetime) ✓
- **c_7**: Reader pushes 1, never popped (program lifetime) ✓
- **c_10**: Reader pushes Wt, never popped (program lifetime) ✓
- **c_12**: Reader pushes Wt, never popped (program lifetime) ✓
- **c_16**: Compute pushes Ht*Wt, Writer pops Ht*Wt ✓

## What Needs to be Implemented (Stage 7)

The compute kernel stub needs to be replaced with the full 11-phase pipeline:

### Pre-loop (ONCE at program start)
- **Pre-Phase 0a**: Tilize gamma (c_10 → c_11, Wt pages → Wt tiles)
- **Pre-Phase 0b**: Tilize beta (c_12 → c_13, Wt pages → Wt tiles)

### Per tile-row loop (Ht iterations)
1. **Phase 1**: Tilize input (c_0 → c_1, Wt pages → Wt tiles)
2. **Phase 2**: Reduce mean (c_1 + c_2 → c_3, PERSISTENT mode)
3. **Phase 3**: Broadcast subtract (c_1 - c_3 → c_4, COL broadcast, pop c_1 and c_3)
4. **Phase 4**: Square (c_4 → c_5, PERSISTENT mode on c_4)
5. **Phase 5**: Reduce variance (c_5 + c_2 → c_6, STREAMING mode)
6-7. **Phases 6-7**: Add epsilon + rsqrt (c_6 + c_7 → c_8, pop c_6, never pop c_7)
8. **Phase 8**: Broadcast multiply rsqrt (c_4 * c_8 → c_9, COL broadcast, pop c_4 and c_8)
10. **Phase 10**: Broadcast multiply gamma (c_9 * c_11 → c_14, ROW broadcast, pop c_9, never pop c_11)
11. **Phase 11**: Broadcast add beta (c_14 + c_13 → c_9, ROW broadcast, pop c_14, never pop c_13)
9. **Phase 9**: Untilize (c_9 → c_16, Wt tiles → Wt pages, pop c_9)

### Key Implementation Details for Stage 7

1. **Use kernel helper libraries** (already used in reference standardize_w_rm):
   - `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp` - tilize()
   - `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp` - untilize()
   - `ttnn/cpp/ttnn/kernel_lib/reduce_helpers.hpp` - reduce<SUM, REDUCE_ROW, mode>()
   - `ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp` - sub(), mul(), add()

2. **CB policies**:
   - PreloadedPopAtEnd: for c_1 in Phase 3, c_4 in Phase 8
   - PreloadedNoPop: for c_4 in Phase 4 (persists to Phase 8)
   - WaitUpfrontPopAtEnd: for most broadcast operations

3. **Broadcast dimensions**:
   - COL broadcast: for mean subtract (Phase 3) and rsqrt multiply (Phase 8) - REDUCE_ROW produces column-shaped output
   - ROW broadcast: for gamma multiply (Phase 10) and beta add (Phase 11) - gamma/beta have shape [1, Wt]

4. **Phases 6-7 (add epsilon + rsqrt)**: Combined, no helper (use DST registers directly)

5. **Reference implementation**: See `standardize_w_rm_compute.cpp` for Phases 1-9 patterns

## Test Results
```
PASSED test_kernels_compile_at_runtime
PASSED test_program_executes_without_hang
```

## Next Agent
**ttnn-kernel-writer** should now implement Stage 7 (correct kernel computation logic).

## Commit
Committed as: `6631bf93aa` - [ttnn-factory-builder] stage 6: stub kernels for layer_norm_w_rm
