# Block Sharded Depthwise Conv2D Implementation Plan

## Status Bar
```
[■■■■■■■■□□□] Step 7 - COMPLETED ✓
```

## Executive Summary

### What We Need to Implement
BLOCK_SHARDED memory layout support for depthwise convolution, where:
- **Rows**: Different cores process different spatial positions (like HEIGHT_SHARDED)
- **Columns**: Different cores process different channel slices (like WIDTH_SHARDED)

This combines the spatial distribution of HEIGHT_SHARDED with the channel distribution of WIDTH_SHARDED.

### Key Architecture Differences from WIDTH_SHARDED

| Aspect | WIDTH_SHARDED | BLOCK_SHARDED |
|--------|---------------|---------------|
| Core organization | 1D row of cores | 2D grid of cores |
| Spatial positions | All cores same | Distributed across rows |
| Channels | Distributed across all cores | Distributed across columns |
| Weight prep | Divide by num_cores | Divide by num_cores_c (columns) |
| Weight distribution | Each core reads DRAM | First row reads, multicasts down |

### Implementation Tasks
1. **Weight Preparation** - Modify to divide by `num_cores_c` instead of `num_cores`
2. **Weight Distribution** - Implement 2D multicast pattern:
   - First row cores: Read from DRAM (like WIDTH_SHARDED), multicast to column
   - Other row cores: Receive weights via multicast (like HEIGHT_SHARDED receivers)

---

## Build & Test Commands

```bash
# Build
./build_metal.sh --release

# Test (with 15s timeout)
source python_env/bin/activate && \
TT_METAL_CLEAR_L1=0 \
TT_METAL_DPRINT_ONE_FILE_PER_RISC=1 \
TT_METAL_DPRINT_CORES="(0,0)" \
timeout 15 pytest tests/ttnn/nightly/unit_tests/operations/conv/test_conv2d.py::test_groups_vs_pool2 -v

# Recovery if hung
tt-smi -r 1
```

---

## Important Rules
1. **DO NOT proceed to the next step until the current step is COMPLETED and VERIFIED**
2. **Only the user can trigger moving to the next step**
3. **After each step, document key takeaways in the "Completed Steps" section**
4. **If test hangs (>15s), recover with: `tt-smi -r 1`**

---

## Reference Files

### From WIDTH_SHARDED Implementation
- **Weight prep**: `ttnn/cpp/ttnn/operations/conv/conv2d/prepare_conv2d_weights.cpp` (~line 1252-1382)
- **Program factory WIDTH_SHARDED path**: `conv2d_op_depthwise_program_factory.cpp` (~line 614-670)
- **Reader kernel**: `ttnn/cpp/ttnn/operations/pool/generic/device/kernels/dataflow/reader_pool_2d.cpp` (~line 528-571)

### Classic Conv2D Block Sharded Kernels (Reference)
- **Sender**: `writer_tiled_out_2d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp`
- **Receiver**: `writer_tiled_out_2d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp`

---

## Step 1: Baseline - Verify HEIGHT_SHARDED Works

### Objective
Confirm the current HEIGHT_SHARDED depthwise conv2d implementation works with a test case that would be similar to block sharded setup.

### Changes Required
**IMPORTANT**: Do NOT read the entire test file (it's ~5400 lines, ~65K tokens). Instead:
1. Use `Grep` to locate the `test_groups_vs_pool2` function
2. Use `Read` with `offset` and `limit` to read ONLY that function and its parametrize decorator

Modify `tests/ttnn/nightly/unit_tests/operations/conv/test_conv2d.py` parametrization for `test_groups_vs_pool2`:

```python
@pytest.mark.parametrize(
    "batch, input_channels, output_channels, input_height, input_width, groups, kernel, stride, padding, dilation, shard_layout, dtype, weights_dtype, bias_dtype, activation, enable_act_double_buffer, enable_weight_double_buffer",
    (
        # Step 1: Baseline HEIGHT_SHARDED test
        (1, 64, 64, 8, 8, 64, (3, 3), (1, 1), (1, 1), (1, 1), ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.bfloat16, ttnn.bfloat16, ttnn.bfloat16, None, False, False),
    ),
)
```

### Test Verification
- [x] Test passes with HEIGHT_SHARDED
- [x] Output matches PyTorch reference (PCC > 0.99)

### Key Takeaways (fill after completion)
```
Step 1 Takeaways:
- HEIGHT_SHARDED baseline test PASSED successfully
- Test configuration: 64 channels, 8x8 spatial, 3x3 kernel, batch=1
- PCC = 0.9998 (exceeds 0.99 threshold)
- Execution time: 0.89s (call time)
- Sharding confirmed: height=true, block=false, width=false
- Weight preparation used HEIGHT_SHARDED 2D depthwise layout correctly
- Test completed without hanging (well under 15s timeout)
```

---

## Step 2: Add BLOCK_SHARDED Test Case (Expected to Fail)

### Objective
Add BLOCK_SHARDED test case to see current failure mode.

### Changes Required
**IMPORTANT**: Use `Grep` to locate the function, then `Read` with `offset`/`limit` to read only that section.

Add BLOCK_SHARDED case to parametrization:

```python
@pytest.mark.parametrize(
    "batch, input_channels, output_channels, input_height, input_width, groups, kernel, stride, padding, dilation, shard_layout, dtype, weights_dtype, bias_dtype, activation, enable_act_double_buffer, enable_weight_double_buffer",
    (
        # Step 2: BLOCK_SHARDED test (expected to fail initially)
        # This will create a 2D grid of cores (e.g., 2x2 = 4 cores)
        # - 2 rows for spatial distribution
        # - 2 columns for channel distribution (32 ch per column)
        (1, 64, 64, 8, 8, 64, (3, 3), (1, 1), (1, 1), (1, 1), ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.bfloat16, ttnn.bfloat16, ttnn.bfloat16, None, False, False),
    ),
)
```

### Test Verification
- [x] Document the error/failure mode
- [x] Identify what code path fails

### Key Takeaways (fill after completion)
```
Step 2 Takeaways:
- BLOCK_SHARDED test FAILED with PCC = 0.089 (threshold: 0.99)
- Program factory correctly detects BLOCK_SHARDED: "height=false, block=true, width=false"
- Input tensor configured correctly with BLOCK_SHARDED on 2x2 grid: [(x=0,y=0) - (x=1,y=1)]
- Shard shape [60, 32] and [32, 32] are correct for BLOCK_SHARDED
- **ROOT CAUSE**: Weight preparation uses HEIGHT_SHARDED layout instead of BLOCK_SHARDED
  - Log shows: "Using HEIGHT_SHARDED 2D depthwise layout"
  - Weights are prepared for HEIGHT_SHARDED but program runs in BLOCK_SHARDED mode
- Weight prep code (prepare_conv2d_weights.cpp) doesn't detect BLOCK_SHARDED properly
- Next step: Fix weight preparation to handle BLOCK_SHARDED case
```

---

## Step 3: Add BLOCK_SHARDED Detection and Checkpoint

### Objective
Add detection for BLOCK_SHARDED memory layout in the depthwise program factory.

### Changes Required
File: `ttnn/cpp/ttnn/operations/conv/conv2d/device/conv2d_op_depthwise_program_factory.cpp`

The `is_block_sharded` variable already exists (line ~318). Add a checkpoint:

```cpp
// After the existing is_width_sharded and is_block_sharded checks (~line 610-618)
if (is_block_sharded) {
    TT_FATAL(false, "BLOCK_SHARDED depthwise conv2d not yet implemented - Step 3 checkpoint");
}
```

### Test Verification
- [x] Build succeeds
- [x] Test fails with our custom "BLOCK_SHARDED not yet implemented" message
- [x] Confirms BLOCK_SHARDED path is being detected

### Key Takeaways (fill after completion)
```
Step 3 Takeaways:
- Checkpoint successfully added at line 674-678 in conv2d_op_depthwise_program_factory.cpp
- Test hits checkpoint with message: "BLOCK_SHARDED depthwise conv2d not yet implemented - Step 3 checkpoint"
- TT_FATAL triggered at correct location (conv2d_op_depthwise_program_factory.cpp:678)
- Confirms BLOCK_SHARDED detection works correctly using is_block_sharded variable
- Test fails cleanly with RuntimeError (as expected)
- Ready to proceed with weight preparation fix in Step 4
```

---

## Step 4: Create Block Sharded Weight Preparation Function

### Objective
Add the weight preparation function for block sharded case. This is similar to WIDTH_SHARDED but divides by `num_cores_c` (number of channel columns) instead of `num_cores` (total cores).

### Background

**WIDTH_SHARDED**: All cores are in a single row, each processing different channels:
```
Core 0: channels 0-31    Core 1: channels 32-63
        ↓                        ↓
     All spatial              All spatial
     positions                positions
```

**BLOCK_SHARDED**: 2D grid where columns process different channels:
```
          col 0 (ch 0-31)    col 1 (ch 32-63)
row 0:    Core (0,0)         Core (0,1)
          spatial 0-31       spatial 0-31

row 1:    Core (1,0)         Core (1,1)
          spatial 32-63      spatial 32-63
```

Cores in the same column need the **SAME** weights (same channels), so weight prep should:
- Divide channels by `num_cores_c` (number of columns), not total cores
- Create the same per-column weight layout as WIDTH_SHARDED

### Changes Required

File: `ttnn/cpp/ttnn/operations/conv/conv2d/prepare_conv2d_weights.cpp`

Modify the WIDTH_SHARDED weight prep wiring to also handle BLOCK_SHARDED:

```cpp
// Find the existing WIDTH_SHARDED wiring (~line 1869-1883)
// Modify to also handle BLOCK_SHARDED:

if (is_depthwise) {
    TensorMemoryLayout shard_layout = conv_config.shard_layout.value_or(TensorMemoryLayout::HEIGHT_SHARDED);

    if (shard_layout == TensorMemoryLayout::WIDTH_SHARDED ||
        shard_layout == TensorMemoryLayout::BLOCK_SHARDED) {

        // For WIDTH_SHARDED: all cores process different channels
        // For BLOCK_SHARDED: only columns process different channels
        uint32_t num_channel_shards = shard_layout == TensorMemoryLayout::BLOCK_SHARDED
            ? params.input_parallel_config->num_cores_c  // Number of columns
            : params.input_parallel_config->grid.num_cores();  // Total cores

        log_info(tt::LogOp, "Using {} depthwise weight prep with {} channel shards (cores_c={})",
                 shard_layout == TensorMemoryLayout::BLOCK_SHARDED ? "BLOCK_SHARDED" : "WIDTH_SHARDED",
                 num_channel_shards,
                 params.input_parallel_config->num_cores_c);

        weight_tensor_ = convert_conv_weight_tensor_to_2d_depthwise_layout_width_sharded(
            weight_tensor_, num_channel_shards, weight_tensor_.dtype());
    } else {
        // HEIGHT_SHARDED: all cores use same weights
        weight_tensor_ = convert_conv_weight_tensor_to_2d_depthwise_layout(
            weight_tensor_, weight_tensor_.dtype());
    }
}
```

### Test Verification
- [x] Build succeeds
- [x] Log message shows correct number of channel shards
- [x] Still fails at program factory checkpoint (Step 3)

### Key Takeaways (fill after completion)
```
Step 4 Takeaways:
- Modified prepare_conv2d_weights.cpp (~line 2061) to detect BLOCK_SHARDED alongside WIDTH_SHARDED
- Uses get_num_cores_channels_from_parallel_config() which correctly returns:
  - For WIDTH_SHARDED: total cores
  - For BLOCK_SHARDED (ROW_MAJOR): grid_size.x (number of columns)
- Weight prep log shows: "Using BLOCK_SHARDED depthwise weight prep with 2 channel shards"
- Weight layout correctly calculated: 64 channels / 2 shards = 32 ch/core, padded to 32
- Output shape [32, 64] is correct (32 rows for padded kernel positions, 64 cols for all channels)
- Test still hits Step 3 checkpoint as expected
- Ready to proceed with weight distribution in Step 5
```

---

## Step 5: Implement Block Sharded Weight Distribution in Program Factory

### Objective
Implement the 2D multicast pattern for BLOCK_SHARDED weight distribution:
- First row cores read their channel slice from DRAM (like WIDTH_SHARDED senders)
- First row cores multicast to all cores in their column (like classic conv2d block sharded)

### Background: 2D Multicast Pattern

In BLOCK_SHARDED with a 2x2 grid:
```
          col 0             col 1
row 0:  Core(0,0)          Core(0,1)
        SENDER for col 0   SENDER for col 1
        reads ch 0-31      reads ch 32-63
        from DRAM          from DRAM
        ↓ mcast            ↓ mcast

row 1:  Core(1,0)          Core(1,1)
        RECEIVER           RECEIVER
        gets ch 0-31       gets ch 32-63
        from Core(0,0)     from Core(0,1)
```

Each column has ONE sender (row 0) and multiple receivers (rows 1+).
Each sender reads different weight tiles (different channels) and multicasts to its column.

### Changes Required

File: `ttnn/cpp/ttnn/operations/conv/conv2d/device/conv2d_op_depthwise_program_factory.cpp`

Replace the Step 3 checkpoint with actual implementation:

```cpp
} else if (is_block_sharded) {
    // ============================================================
    // BLOCK_SHARDED: 2D multicast pattern
    // - First row cores read from DRAM and multicast to column
    // - Other row cores receive from first row
    // ============================================================

    // Get grid dimensions
    uint32_t num_cores_c = parallelization_config.num_cores_c_in;  // Number of columns (channel shards)
    uint32_t num_cores_r = num_cores / num_cores_c;  // Number of rows (spatial shards)

    // Calculate per-column weight shard size (same as WIDTH_SHARDED but per column)
    uint32_t channels_per_column = in_c_per_shard_ceil;  // Already calculated for this
    uint32_t padded_ch_per_column = tt::round_up(channels_per_column, tt::constants::TILE_WIDTH);
    uint32_t padded_kernel_pos = tt::round_up(kernel_h * kernel_w, tt::constants::TILE_WIDTH);
    uint32_t shard_ntiles =
        (padded_ch_per_column / tt::constants::TILE_WIDTH) * (padded_kernel_pos / tt::constants::TILE_WIDTH);
    uint32_t weight_tile_nbytes = tt::tile_size(params.data_format);
    uint32_t shard_size_bytes = shard_ntiles * weight_tile_nbytes;

    log_info(tt::LogOp, "BLOCK_SHARDED weights: {}x{} grid, {} ch/col, {} tiles/col, {} bytes/col",
             num_cores_r, num_cores_c, channels_per_column, shard_ntiles, shard_size_bytes);

    // Set up each core
    for (uint32_t core_idx = 0; core_idx < num_cores; core_idx++) {
        CoreCoord core = all_cores[core_idx];

        // Determine row and column in the grid
        uint32_t col_idx = core_idx % num_cores_c;
        uint32_t row_idx = core_idx / num_cores_c;

        bool is_sender = (row_idx == 0);  // First row cores are senders

        if (is_sender) {
            // ============================================================
            // SENDER: Read from DRAM and multicast to column
            // ============================================================

            // Calculate starting tile_id for this column's channel slice
            uint32_t start_tile_id = col_idx * shard_ntiles;

            // Get physical coordinates for multicast destination (entire column)
            // Multicast from row 0 to row (num_cores_r - 1) in this column
            CoreCoord col_start_logical(col_idx, 0);
            CoreCoord col_end_logical(col_idx, num_cores_r - 1);
            CoreCoord col_start_physical = device->worker_core_from_logical_core(col_start_logical);
            CoreCoord col_end_physical = device->worker_core_from_logical_core(col_end_logical);

            // Number of receivers = all rows except sender (row 0)
            uint32_t num_dests = num_cores_r - 1;
            uint32_t num_mcast_cores = num_cores_r - 1;

            std::vector<uint32_t> sender_args = {
                static_cast<uint32_t>(weight_buffer_addr),  // 0: weight_addr (SAME base for all senders)
                1,                                          // 1: is_sender = true
                col_start_physical.x,                       // 2: weights_mcast_dest_noc_start_x
                col_start_physical.y,                       // 3: weights_mcast_dest_noc_start_y
                col_end_physical.x,                         // 4: weights_mcast_dest_noc_end_x
                col_end_physical.y,                         // 5: weights_mcast_dest_noc_end_y
                num_dests,                                  // 6: weights_mcast_num_dests
                num_mcast_cores,                            // 7: weights_mcast_num_cores
                weights_mcast_sender_semaphore_id,          // 8: sender semaphore
                weights_mcast_receiver_semaphore_id,        // 9: receiver semaphore
                start_tile_id                               // 10: starting tile_id for this column
            };
            SetRuntimeArgs(program, reader0_kernel, core, sender_args);

            log_info(tt::LogOp, "BLOCK_SHARDED Sender[{}] col={}: tile_id={}, mcast to {} receivers",
                     core_idx, col_idx, start_tile_id, num_dests);

        } else {
            // ============================================================
            // RECEIVER: Wait for multicast from column's sender (row 0)
            // ============================================================

            // Find the sender for this column (row 0, same column)
            CoreCoord sender_logical(col_idx, 0);
            CoreCoord sender_physical = device->worker_core_from_logical_core(sender_logical);

            std::vector<uint32_t> receiver_args = {
                static_cast<uint32_t>(weight_buffer_addr),  // 0: weight_addr (unused but consistent)
                0,                                          // 1: is_sender = false
                sender_physical.x,                          // 2: weights_mcast_sender_noc_x
                sender_physical.y,                          // 3: weights_mcast_sender_noc_y
                weights_mcast_sender_semaphore_id,          // 4: sender semaphore
                weights_mcast_receiver_semaphore_id         // 5: receiver semaphore
            };
            SetRuntimeArgs(program, reader0_kernel, core, receiver_args);

            log_info(tt::LogOp, "BLOCK_SHARDED Receiver[{}] col={} row={}: waiting for sender at col={}",
                     core_idx, col_idx, row_idx, col_idx);
        }
    }

    log_info(tt::LogOp, "BLOCK_SHARDED: {} senders (row 0), {} receivers per column",
             num_cores_c, num_cores_r - 1);

} else {
    // HEIGHT_SHARDED: existing multicast code
    // ...
}
```

### Key Considerations

1. **Sender tile_id calculation**: Each sender (first row) reads different tiles based on column index
2. **Multicast direction**: Column-wise (vertical) multicast, not row-wise
3. **Receiver identification**: All cores in row > 0 are receivers

### Test Verification
- [x] Build succeeds
- [x] Test runs without hanging
- [x] Logs show correct sender/receiver setup
- [x] Verify senders read correct tile IDs

### Key Takeaways (fill after completion)
```
Step 5 Takeaways:
- BLOCK_SHARDED weight distribution implemented successfully!
- Modified kernel (reader_pool_2d.cpp) to always read start_tile_id from arg 10
- Updated HEIGHT_SHARDED sender to pass arg 10 = 0 for consistency
- BLOCK_SHARDED uses column-wise multicast pattern:
  - 2x2 grid: 2 columns (channel shards) × 2 rows (spatial shards)
  - Sender[0] col=0: tile_id=0, multicasts to row 1
  - Sender[1] col=1: tile_id=1, multicasts to row 1
- Each sender reads from DRAM with offset based on column index
- Multicast destination coordinates are vertical (same x, different y)
- Test PASSED with PCC = 0.9998 (threshold: 0.99)
- Execution time: 0.49s call, 0.82s total
```

---

## Step 6: Verify Numerical Correctness

### Objective
Verify the output matches PyTorch reference with block sharded implementation.

### Test Configuration
```python
# Test case: 64 channels, 2x2 grid, 3x3 kernel
(1, 64, 64, 8, 8, 64, (3, 3), (1, 1), (1, 1), (1, 1),
 ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.bfloat16, ttnn.bfloat16, ttnn.bfloat16,
 None, False, False)
```

### Test Verification
- [x] Output shape matches reference
- [x] PCC >= 0.99 (threshold)
- [x] Test PASSES

### Key Takeaways (fill after completion)
```
Step 6 Takeaways:
- Numerical correctness VERIFIED in Step 5 test run
- PCC = 0.9998 (well above 0.99 threshold)
- Output shape: [1, 1, 64, 64] matches PyTorch reference
- Block sharded depthwise conv2d produces correct results
- Weight distribution via column multicast working correctly
- All 4 cores (2x2 grid) processing correct data
```

---

## Step 7: Expand Test Coverage

### Objective
Test with more configurations to ensure robustness.

### Changes Required
Uncommented existing BLOCK_SHARDED test cases from `test_groups_vs_pool2`:

```python
# Passing tests:
(1, 64, 64, 8, 8, 64, (3, 3), (1, 1), (1, 1), (1, 1), BLOCK_SHARDED, ...)      # Basic test
(1, 32, 32, 4, 8, 32, (3, 3), (1, 1), (1, 1), (1, 1), BLOCK_SHARDED, gelu)     # With gelu activation
(10, 576, 576, 14, 14, 576, (3, 3), ..., BLOCK_SHARDED, relu6)                 # MobileNetV2 - batch 10!
(1, 672, 672, 14, 14, 672, (5, 5), (1, 1), ..., BLOCK_SHARDED)                 # EfficientNet-B0 5x5 kernel
(1, 672, 672, 14, 14, 672, (5, 5), (2, 2), ..., BLOCK_SHARDED)                 # EfficientNet-B0 with stride
```

### Test Verification
- [x] All 5 test cases pass
- [x] Different channel counts work (32, 64, 576, 672)
- [x] Activation functions work (gelu, relu6)
- [x] Different kernel sizes work (3x3, 5x5)
- [x] Different batch sizes work (1, 10)
- [x] Stride variations work (1x1, 2x2)

### Key Takeaways (fill after completion)
```
Step 7 Takeaways:
- 5 BLOCK_SHARDED tests pass with various configurations
- Verified configurations:
  1. 64 ch, 8x8, 3x3 kernel, 2x2 grid - PASSED
  2. 32 ch, 4x8, 3x3 kernel, gelu activation - PASSED
  3. 576 ch, 14x14, 3x3 kernel, relu6, batch=10 (MobileNetV2) - PASSED
  4. 672 ch, 14x14, 5x5 kernel, 7x2 grid (EfficientNet-B0) - PASSED
  5. 672 ch, 14x14, 5x5 kernel, stride 2x2 (EfficientNet-B0) - PASSED
- All PCCs > 0.99 threshold
- Some larger configurations fail due to memory issues (not block sharded specific)
- Block sharded works correctly for EfficientNet-B0 and MobileNetV2 workloads
```

---

## Completed Steps Log

### Step 1: [COMPLETED ✓]
```
Status: PASSED - Test completed successfully with PCC = 0.9998
Takeaways:
- HEIGHT_SHARDED baseline test PASSED successfully
- Test configuration: 64 channels, 8x8 spatial, 3x3 kernel, batch=1
- PCC = 0.9998 (exceeds 0.99 threshold)
- Execution time: 0.89s (call time)
- Sharding confirmed: height=true, block=false, width=false
- Weight preparation used HEIGHT_SHARDED 2D depthwise layout correctly
- Test completed without hanging (well under 15s timeout)
```

### Step 2: [COMPLETED ✓]
```
Status: FAILED - Test failed with PCC = 0.089 (expected failure)
Takeaways:
- BLOCK_SHARDED test FAILED with PCC = 0.089 (threshold: 0.99)
- Program factory correctly detects BLOCK_SHARDED: "height=false, block=true, width=false"
- Input tensor configured correctly with BLOCK_SHARDED on 2x2 grid: [(x=0,y=0) - (x=1,y=1)]
- Shard shape [60, 32] and [32, 32] are correct for BLOCK_SHARDED
- **ROOT CAUSE**: Weight preparation uses HEIGHT_SHARDED layout instead of BLOCK_SHARDED
  - Log shows: "Using HEIGHT_SHARDED 2D depthwise layout"
  - Weights are prepared for HEIGHT_SHARDED but program runs in BLOCK_SHARDED mode
- Weight prep code (prepare_conv2d_weights.cpp) doesn't detect BLOCK_SHARDED properly
- Next step: Fix weight preparation to handle BLOCK_SHARDED case
```

### Step 3: [COMPLETED ✓]
```
Status: CHECKPOINT WORKING - Test fails cleanly at checkpoint with expected message
Takeaways:
- Checkpoint successfully added at line 674-678 in conv2d_op_depthwise_program_factory.cpp
- Test hits checkpoint with message: "BLOCK_SHARDED depthwise conv2d not yet implemented - Step 3 checkpoint"
- TT_FATAL triggered at correct location (conv2d_op_depthwise_program_factory.cpp:678)
- Confirms BLOCK_SHARDED detection works correctly using is_block_sharded variable
- Test fails cleanly with RuntimeError (as expected)
- Ready to proceed with weight preparation fix in Step 4
```

### Step 4: [COMPLETED ✓]
```
Status: PASSED - Weight preparation correctly handles BLOCK_SHARDED case
Takeaways:
- Modified prepare_conv2d_weights.cpp (~line 2061) to detect BLOCK_SHARDED alongside WIDTH_SHARDED
- Uses get_num_cores_channels_from_parallel_config() which correctly returns:
  - For WIDTH_SHARDED: total cores
  - For BLOCK_SHARDED (ROW_MAJOR): grid_size.x (number of columns)
- Weight prep log shows: "Using BLOCK_SHARDED depthwise weight prep with 2 channel shards"
- Weight layout correctly calculated: 64 channels / 2 shards = 32 ch/core, padded to 32
- Output shape [32, 64] is correct (32 rows for padded kernel positions, 64 cols for all channels)
- Test still hits Step 3 checkpoint as expected
- Ready to proceed with weight distribution in Step 5
```

### Step 5: [COMPLETED ✓]
```
Status: PASSED - Test passed with PCC = 0.9998
Takeaways:
- BLOCK_SHARDED weight distribution implemented successfully!
- Modified kernel (reader_pool_2d.cpp) to always read start_tile_id from arg 10
- Updated HEIGHT_SHARDED sender to pass arg 10 = 0 for consistency
- BLOCK_SHARDED uses column-wise multicast pattern:
  - 2x2 grid: 2 columns (channel shards) × 2 rows (spatial shards)
  - Sender[0] col=0: tile_id=0, multicasts to row 1
  - Sender[1] col=1: tile_id=1, multicasts to row 1
- Each sender reads from DRAM with offset based on column index
- Multicast destination coordinates are vertical (same x, different y)
- Test PASSED with PCC = 0.9998 (threshold: 0.99)
- Execution time: 0.49s call, 0.82s total
```

### Step 6: [COMPLETED ✓]
```
Status: PASSED - Numerical correctness verified with PCC = 0.9998
Takeaways:
- Numerical correctness VERIFIED in Step 5 test run
- PCC = 0.9998 (well above 0.99 threshold)
- Output shape: [1, 1, 64, 64] matches PyTorch reference
- Block sharded depthwise conv2d produces correct results
- Weight distribution via column multicast working correctly
- All 4 cores (2x2 grid) processing correct data
```

### Step 7: [COMPLETED ✓]
```
Status: PASSED - 5 BLOCK_SHARDED tests pass with various configurations
Takeaways:
- Verified configurations:
  1. 64 ch, 8x8, 3x3 kernel, 2x2 grid - PASSED
  2. 32 ch, 4x8, 3x3 kernel, gelu activation - PASSED
  3. 576 ch, 14x14, 3x3 kernel, relu6, batch=10 (MobileNetV2) - PASSED
  4. 672 ch, 14x14, 5x5 kernel, 7x2 grid (EfficientNet-B0) - PASSED
  5. 672 ch, 14x14, 5x5 kernel, stride 2x2 (EfficientNet-B0) - PASSED
- All PCCs > 0.99 threshold
- Some larger configurations fail due to memory issues (not block sharded specific)
- Block sharded works correctly for EfficientNet-B0 and MobileNetV2 workloads
```

---

## Debugging Tips

### 1. Use Integer Weights for Easy Debugging

In `test_groups_vs_pool2`, uncomment the integer weights section:

```python
for out_ch in range(conv_weight_shape[0]):
    for in_ch in range(conv_weight_shape[1]):
        for kh in range(conv_weight_shape[2]):
            for kw in range(conv_weight_shape[3]):
                stick_id = kh * kernel[1] + kw + 1
                torch_weight_tensor[out_ch, in_ch, kh, kw] = stick_id
```

### 2. Verify Core Grid Layout

For BLOCK_SHARDED, verify the logical-to-physical core mapping:

```cpp
// Add debug logging
for (uint32_t core_idx = 0; core_idx < num_cores; core_idx++) {
    CoreCoord core = all_cores[core_idx];
    CoreCoord physical = device->worker_core_from_logical_core(core);
    uint32_t col = core_idx % num_cores_c;
    uint32_t row = core_idx / num_cores_c;
    log_info(tt::LogOp, "Core[{}]: logical({},{}), physical({},{}), grid_pos=({},{})",
             core_idx, core.x, core.y, physical.x, physical.y, col, row);
}
```

### 3. Verify Multicast Coordinates

For column-wise multicast, ensure:
- Start and end coordinates define a vertical column
- Same x coordinate for start and end
- y coordinates span from 0 to num_cores_r-1

---

## Architecture Diagram

```
BLOCK_SHARDED 2x2 Grid:

    +---------------+---------------+
    | Core(0,0)     | Core(0,1)     |  ROW 0: SENDERS
    | SENDER        | SENDER        |
    | ch 0-31       | ch 32-63      |
    | reads DRAM    | reads DRAM    |
    | tile_id=0     | tile_id=1     |
    |     ↓ mcast   |     ↓ mcast   |
    +---------------+---------------+
    | Core(1,0)     | Core(1,1)     |  ROW 1+: RECEIVERS
    | RECEIVER      | RECEIVER      |
    | ch 0-31       | ch 32-63      |
    | from Core0,0  | from Core0,1  |
    +---------------+---------------+

    Column 0        Column 1
    (spatial 0-N/2) (spatial N/2-N)
```

---

## Quick Reference

### File Locations
- Weight prep: `ttnn/cpp/ttnn/operations/conv/conv2d/prepare_conv2d_weights.cpp`
- Program factory: `ttnn/cpp/ttnn/operations/conv/conv2d/device/conv2d_op_depthwise_program_factory.cpp`
- Reader kernel: `ttnn/cpp/ttnn/operations/pool/generic/device/kernels/dataflow/reader_pool_2d.cpp`

### Key Variables to Track
- `num_cores_c`: Number of columns (channel distribution)
- `num_cores_r`: Number of rows (spatial distribution)
- `in_c_per_shard_ceil`: Channels per column
- `shard_ntiles`: Weight tiles per column
- `start_tile_id`: Starting tile for each column's sender
