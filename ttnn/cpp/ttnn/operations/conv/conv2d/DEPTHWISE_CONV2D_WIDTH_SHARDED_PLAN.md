# Width Sharded Depthwise Conv2D Implementation Plan

## Status Bar
```
[■■■■■■□□□□□] Step 5 of 11 - COMPLETED
```

## Important Rules
1. **DO NOT proceed to the next step until the current step is COMPLETED and VERIFIED**
2. **Only the user can trigger moving to the next step**
3. **After each step, document key takeaways in the "Completed Steps" section**
4. **If test hangs (>15s), recover with: `tt-smi -r 1`**

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

## Step 1: Baseline - Verify HEIGHT_SHARDED Works

### Objective
Confirm the current HEIGHT_SHARDED depthwise conv2d implementation works with our simplified test case.

### Changes Required
Modify `tests/ttnn/nightly/unit_tests/operations/conv/test_conv2d.py` parametrization for `test_groups_vs_pool2`:

```python
@pytest.mark.parametrize(
    "batch, input_channels, output_channels, input_height, input_width, groups, kernel, stride, padding, dilation, shard_layout, dtype, weights_dtype, bias_dtype, activation, enable_act_double_buffer, enable_weight_double_buffer",
    (
        # Step 1: Baseline HEIGHT_SHARDED test (64 channels, simple case)
        (1, 64, 64, 8, 8, 64, (3, 3), (1, 1), (1, 1), (1, 1), ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.bfloat16, ttnn.bfloat16, ttnn.bfloat16, None, False, False),
    ),
)
```

### Test Verification
- [x] Test passes with HEIGHT_SHARDED
- [x] Output matches PyTorch reference

### Key Takeaways (fill after completion)
```
Step 1 Takeaways:
- HEIGHT_SHARDED baseline works with PCC = 0.9998038038213132 (threshold 0.99)
- Uses 2 cores with grid={(0,0)-(1,0)}, shard shape {60, 64} input / {32, 64} output
- 2D depthwise layout conversion confirmed: "Using 2D depthwise layout conversion for groups=64, channels=64, kernel=3x3"
```

---

## Step 2: Add WIDTH_SHARDED Test Case (Expected to Fail)

### Objective
Add WIDTH_SHARDED test case to see current failure mode.

### Changes Required
Add WIDTH_SHARDED case to parametrization:

```python
@pytest.mark.parametrize(
    "batch, input_channels, output_channels, input_height, input_width, groups, kernel, stride, padding, dilation, shard_layout, dtype, weights_dtype, bias_dtype, activation, enable_act_double_buffer, enable_weight_double_buffer",
    (
        # Step 2: WIDTH_SHARDED test (expected to fail initially)
        (1, 64, 64, 8, 8, 64, (3, 3), (1, 1), (1, 1), (1, 1), ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.bfloat16, ttnn.bfloat16, ttnn.bfloat16, None, False, False),
    ),
)
```

### Test Verification
- [x] Document the error/failure mode
- [x] Identify what code path fails

### Key Takeaways (fill after completion)
```
Step 2 Takeaways:
- Error message: PCC = 0.0895812379007859 (threshold 0.99) - produces incorrect numerical results
- Failure location: test_conv2d.py:5436 (PCC comparison assertion)
- Root cause: Weight preparation uses HEIGHT_SHARDED layout (all cores same weights via multicast)
  but WIDTH_SHARDED needs per-core weight slices (each core has different channels)
- Sharding config: grid={(0,0)-(1,0)}, shard shape {100,32} input / {64,32} output (32 ch per core)
- Existing weight prep function "Using 2D depthwise layout conversion" is not WIDTH_SHARDED aware
```

---

## Step 3: Add Width Sharded Detection in Program Factory

### Objective
Add detection for WIDTH_SHARDED memory layout in the depthwise program factory.

### Changes Required
File: `ttnn/cpp/ttnn/operations/conv/conv2d/device/conv2d_op_depthwise_program_factory.cpp`

```cpp
// After line ~318 (after is_block_sharded detection)
bool is_width_sharded = (a.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED);

log_info(tt::LogOp, "Depthwise conv2d sharding mode: height={}, block={}, width={}",
         is_height_sharded, is_block_sharded, is_width_sharded);

if (is_width_sharded) {
    TT_FATAL(false, "WIDTH_SHARDED depthwise conv2d not yet implemented - Step 3 checkpoint");
}
```

### Test Verification
- [x] Build succeeds
- [x] Test fails with our custom "WIDTH_SHARDED not yet implemented" message
- [x] Confirms WIDTH_SHARDED path is being detected

### Key Takeaways (fill after completion)
```
Step 3 Takeaways:
- Detection added at line 739-745 in conv2d_op_depthwise_program_factory.cpp
- Log confirms: "Depthwise conv2d sharding mode: height=false, block=false, width=true"
- TT_FATAL correctly triggers: "WIDTH_SHARDED depthwise conv2d not yet implemented - Step 3 checkpoint"
- Note: is_width_sharded variable already existed at line 739, we just added the check
```

---

## Step 4: Create Width Sharded Weight Preparation Function Skeleton

### Objective
Add the skeleton for the width sharded weight preparation function.

### Changes Required
File: `ttnn/cpp/ttnn/operations/conv/conv2d/prepare_conv2d_weights.cpp`

Add after `convert_conv_weight_tensor_to_2d_depthwise_layout` (~line 1236):

```cpp
/*
Width sharded depthwise weight preparation.
Creates per-shard weights with face-by-face layout.
*/
Tensor convert_conv_weight_tensor_to_2d_depthwise_layout_width_sharded(
    const Tensor& conv_weight_tensor,
    uint32_t num_channel_shards,
    DataType output_dtype) {

    log_info(tt::LogOp, "Width sharded weight prep: num_shards={}", num_channel_shards);

    // For now, just call the existing function (will produce wrong results)
    // This is a checkpoint to verify the function is being called
    return convert_conv_weight_tensor_to_2d_depthwise_layout(conv_weight_tensor, output_dtype);
}
```

File: `ttnn/cpp/ttnn/operations/conv/conv2d/prepare_conv2d_weights.hpp`

Add declaration:
```cpp
Tensor convert_conv_weight_tensor_to_2d_depthwise_layout_width_sharded(
    const Tensor& conv_weight_tensor,
    uint32_t num_channel_shards,
    DataType output_dtype);
```

### Test Verification
- [x] Build succeeds
- [x] Function skeleton is in place

### Key Takeaways (fill after completion)
```
Step 4 Takeaways:
- Skeleton function added at line 1242-1252 in prepare_conv2d_weights.cpp
- Declaration added at line 83-87 in prepare_conv2d_weights.hpp
- Currently just wraps existing function (will produce wrong results until Step 6)
```

---

## Step 5: Wire Up Width Sharded Weight Preparation

### Objective
Call the width sharded weight preparation function when WIDTH_SHARDED is detected.

### Changes Required
File: `ttnn/cpp/ttnn/operations/conv/conv2d/prepare_conv2d_weights.cpp`

Find where `convert_conv_weight_tensor_to_2d_depthwise_layout` is called (~line 1853) and modify:

```cpp
if (is_depthwise) {
    TensorMemoryLayout shard_layout = conv_config.shard_layout.value_or(TensorMemoryLayout::HEIGHT_SHARDED);

    if (shard_layout == TensorMemoryLayout::WIDTH_SHARDED) {
        uint32_t num_cores = parallel_config.grid.num_cores();
        log_info(tt::LogOp, "Using WIDTH_SHARDED depthwise weight prep with {} cores", num_cores);
        weight_tensor_ = convert_conv_weight_tensor_to_2d_depthwise_layout_width_sharded(
            weight_tensor_, num_cores, weight_tensor_.dtype());
    } else {
        weight_tensor_ = convert_conv_weight_tensor_to_2d_depthwise_layout(
            weight_tensor_, weight_tensor_.dtype());
    }
}
```

### Test Verification
- [x] Build succeeds
- [x] Log message shows "Using WIDTH_SHARDED depthwise weight prep"
- [x] Still fails at program factory (Step 3 checkpoint)

### Key Takeaways (fill after completion)
```
Step 5 Takeaways:
- Wiring added at lines 1869-1883 in prepare_conv2d_weights.cpp
- Detects WIDTH_SHARDED via params.input_parallel_config->shard_scheme
- Gets num_cores from params.input_parallel_config->grid.num_cores()
- Log confirms: "Using WIDTH_SHARDED depthwise weight prep with 2 cores"
- Correctly routes to skeleton function (which still wraps existing function for now)
```

---

## Step 6: Implement Width Sharded Weight Layout (Core Logic)

### Objective
Implement the actual face-by-face weight layout for width sharded case.

### Changes Required
File: `ttnn/cpp/ttnn/operations/conv/conv2d/prepare_conv2d_weights.cpp`

Replace the skeleton with full implementation:

```cpp
template <typename T>
static Tensor conv_2d_depthwise_weight_layout_width_sharded_helper(
    const Tensor& conv_weight_tensor,
    const ttnn::Shape& original_weight_shape,
    uint32_t num_channel_shards,
    DataType output_dtype) {

    uint32_t out_channels = original_weight_shape[0];
    uint32_t kernel_h = original_weight_shape[2];
    uint32_t kernel_w = original_weight_shape[3];
    uint32_t total_kernel_positions = kernel_h * kernel_w;

    // Per-shard dimensions
    uint32_t channels_per_shard = out_channels / num_channel_shards;

    constexpr uint32_t TILE_SIZE = 32;
    constexpr uint32_t FACE_SIZE = 16;

    // Pad per-shard to tile boundaries
    uint32_t padded_channels_per_shard = ((channels_per_shard + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE;
    uint32_t padded_kernel_positions = ((total_kernel_positions + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE;

    // Total output: all shards concatenated
    uint32_t total_padded_channels = padded_channels_per_shard * num_channel_shards;
    ttnn::Shape output_shape{total_padded_channels, padded_kernel_positions, 1, 1};

    log_info(tt::LogOp, "Width sharded weight layout: channels={}, shards={}, ch_per_shard={}, padded={}",
             out_channels, num_channel_shards, channels_per_shard, padded_channels_per_shard);

    auto compute = [&](const tt::tt_metal::HostBuffer& input_host_buffer) {
        auto input_buffer = tt::tt_metal::host_buffer::get_as<T>(input_host_buffer);
        auto output_buffer = std::vector<T>(output_shape.volume(), static_cast<T>(0));

        uint32_t data_rows_per_stick = (channels_per_shard + FACE_SIZE - 1) / FACE_SIZE;
        uint32_t rows_per_stick_padded = (padded_channels_per_shard + FACE_SIZE - 1) / FACE_SIZE;
        uint32_t shard_size = padded_channels_per_shard * padded_kernel_positions;

        // Process each shard with face-by-face filling
        for (uint32_t shard_idx = 0; shard_idx < num_channel_shards; shard_idx++) {
            uint32_t channel_start = shard_idx * channels_per_shard;
            uint32_t shard_offset = shard_idx * shard_size;
            uint32_t current_absolute_row = 0;

            for (uint32_t kernel_pos = 0; kernel_pos < total_kernel_positions; kernel_pos++) {
                uint32_t kh = kernel_pos / kernel_w;
                uint32_t kw = kernel_pos % kernel_w;

                for (uint32_t stick_row = 0; stick_row < data_rows_per_stick; stick_row++) {
                    uint32_t absolute_row = current_absolute_row + stick_row;
                    uint32_t face_idx = absolute_row / FACE_SIZE;
                    uint32_t row_in_face = absolute_row % FACE_SIZE;

                    uint32_t tile_idx = face_idx / 4;
                    uint32_t face_in_tile = face_idx % 4;

                    uint32_t face_row_offset = (face_in_tile / 2) * FACE_SIZE;
                    uint32_t face_col_offset = (face_in_tile % 2) * FACE_SIZE;
                    uint32_t target_row = face_row_offset + row_in_face;

                    for (uint32_t col = 0; col < FACE_SIZE; col++) {
                        uint32_t local_ch = stick_row * FACE_SIZE + col;
                        if (local_ch >= channels_per_shard) break;

                        uint32_t global_ch = channel_start + local_ch;
                        if (global_ch >= out_channels) break;

                        auto input_idx = tt::tt_metal::compute_flat_indices(
                            ttnn::SmallVector<int>{(int)global_ch, 0, (int)kh, (int)kw},
                            compute_strides(original_weight_shape));
                        T value = input_buffer[input_idx];

                        uint32_t target_col = tile_idx * TILE_SIZE + face_col_offset + col;
                        uint32_t output_idx = shard_offset + target_row * padded_channels_per_shard + target_col;

                        if (output_idx < output_buffer.size()) {
                            output_buffer[output_idx] = value;
                        }
                    }
                }
                current_absolute_row += rows_per_stick_padded;
            }
        }

        return tt::tt_metal::HostBuffer(std::move(output_buffer));
    };

    const TensorSpec output_spec(
        output_shape,
        tt::tt_metal::TensorLayout(output_dtype, tt::tt_metal::PageConfig(Layout::ROW_MAJOR), MemoryConfig{}));
    return convert_tensor<T>(conv_weight_tensor, compute, output_spec);
}

Tensor convert_conv_weight_tensor_to_2d_depthwise_layout_width_sharded(
    const Tensor& conv_weight_tensor,
    uint32_t num_channel_shards,
    DataType output_dtype) {

    const auto& original_shape = conv_weight_tensor.logical_shape();

    const static std::unordered_map<DataType,
        std::function<Tensor(const Tensor&, ttnn::Shape, uint32_t, DataType)>>
        layout_map = {
            {DataType::BFLOAT16, &conv_2d_depthwise_weight_layout_width_sharded_helper<bfloat16>},
            {DataType::FLOAT32, &conv_2d_depthwise_weight_layout_width_sharded_helper<float>},
        };

    output_dtype = ((output_dtype == DataType::BFLOAT8_B) || (output_dtype == DataType::BFLOAT4_B))
                       ? DataType::FLOAT32 : output_dtype;

    return layout_map.at(conv_weight_tensor.dtype())(
        conv_weight_tensor, original_shape, num_channel_shards, output_dtype);
}
```

### Test Verification
- [ ] Build succeeds
- [ ] Weight tensor is created with correct shape
- [ ] Still fails at program factory checkpoint

### Key Takeaways (fill after completion)
```
Step 6 Takeaways:
-
```

---

## Step 7: Modify Reader Kernel for Per-Core Weight Reading

### Objective
Modify the reader kernel (`reader_pool_2d.cpp`) to support WIDTH_SHARDED mode where each core reads its own weight slice from DRAM instead of using multicast.

### Background
- **HEIGHT_SHARDED**: Core 0 reads all weights from DRAM and multicasts to other cores (all cores process same channels, different spatial positions)
- **WIDTH_SHARDED**: Each core reads its own weight slice from DRAM (each core processes different channels, same spatial positions)

### Changes Required
File: `ttnn/cpp/ttnn/operations/pool/generic/device/kernels/dataflow/reader_pool_2d.cpp`

Modify the weight reading logic to handle per-core DRAM reads:

```cpp
// Add a runtime arg or compile-time define for width_sharded mode
// When width_sharded:
//   - Each core reads from weight_addr + (core_id * shard_size_bytes)
//   - Skip multicast sender/receiver logic
//   - All cores act as "readers" for their own slice

// In the weight reading section, add WIDTH_SHARDED path:
#ifdef WIDTH_SHARDED_WEIGHTS
    // Each core reads its own weight slice directly from DRAM
    uint64_t weight_noc_addr = get_noc_addr(weight_addr);
    noc_async_read(weight_noc_addr, weight_cb_addr, weight_size_bytes);
    noc_async_read_barrier();
#else
    // Existing multicast logic for HEIGHT_SHARDED
    if (is_sender) {
        // Read from DRAM and multicast to receivers
        ...
    } else {
        // Wait for multicast from sender
        ...
    }
#endif
```

### Key Considerations
1. The program factory will pass different `weight_addr` to each core (with per-core offset)
2. No semaphore synchronization needed (no multicast)
3. Each core independently reads its weight slice
4. Weight CB size remains the same (per-core shard size)

### Test Verification
- [ ] Build succeeds
- [ ] Kernel compiles with WIDTH_SHARDED_WEIGHTS define
- [ ] Each core can independently read from DRAM

### Key Takeaways (fill after completion)
```
Step 7 Takeaways:
-
```

---

## Step 8: Implement Width Sharded Program Factory - No Multicast

### Objective
Implement the program factory path for WIDTH_SHARDED (set up per-core weight addresses and compile kernel with WIDTH_SHARDED_WEIGHTS).

### Changes Required
File: `ttnn/cpp/ttnn/operations/conv/conv2d/device/conv2d_op_depthwise_program_factory.cpp`

Replace the Step 3 TT_FATAL with actual implementation:

```cpp
if (is_width_sharded) {
    // ============================================================
    // WIDTH SHARDED: Each core reads its own weights from DRAM
    // No multicast - each core has unique channels
    // ============================================================

    uint32_t channels_per_core = output_channels / num_cores;
    uint32_t padded_channels_per_core = tt::round_up(channels_per_core, 32);
    uint32_t padded_kernel_positions = tt::round_up(kernel_h * kernel_w, 32);

    // Calculate weight tile count for this core's shard
    uint32_t shard_ntiles = (padded_channels_per_core * padded_kernel_positions) / (32 * 32);
    uint32_t weight_tile_nbytes = tt::tile_size(params.data_format);
    uint32_t shard_size_bytes = shard_ntiles * weight_tile_nbytes;

    log_info(tt::LogOp, "Width sharded weight distribution:");
    log_info(tt::LogOp, "  num_cores: {}", num_cores);
    log_info(tt::LogOp, "  channels_per_core: {}", channels_per_core);
    log_info(tt::LogOp, "  shard_ntiles: {}", shard_ntiles);
    log_info(tt::LogOp, "  shard_size_bytes: {}", shard_size_bytes);

    // Add WIDTH_SHARDED_WEIGHTS define to reader kernel
    reader_defines["WIDTH_SHARDED_WEIGHTS"] = "1";

    // Each core reads its own weight slice - NO multicast
    for (uint32_t core_idx = 0; core_idx < num_cores; core_idx++) {
        CoreCoord core = all_cores[core_idx];
        uint32_t weight_offset = core_idx * shard_size_bytes;

        std::vector<uint32_t> reader_args = {
            static_cast<uint32_t>(weight_buffer_addr + weight_offset),
            // ... other args
        };

        SetRuntimeArgs(program, reader0_kernel, core, reader_args);
        log_debug(tt::LogOp, "Core {}: weight_offset={}", core_idx, weight_offset);
    }

} else {
    // HEIGHT SHARDED: existing multicast code
    // ... (keep existing code)
}
```

### Test Verification
- [ ] Build succeeds
- [ ] Test runs without hanging
- [ ] Verify log shows per-core weight offsets

### Key Takeaways (fill after completion)
```
Step 8 Takeaways:
-
```

---

## Step 9: Debug Weight Reading in Kernel

### Objective
Verify the reader kernel correctly reads weights for each core.

### Changes Required
Add debug prints to `reader_pool_2d.cpp` in the weight reading section:

```cpp
// In the WIDTH_SHARDED path
DPRINT << "Core reading weights: addr=" << weight_addr
       << " size=" << weight_size_bytes << ENDL();
```

### Test Verification
- [ ] See DPRINT output showing each core's weight address
- [ ] Verify offsets are different for each core
- [ ] Test completes (may have wrong results)

### Key Takeaways (fill after completion)
```
Step 9 Takeaways:
-
```

---

## Step 10: Verify Numerical Correctness

### Objective
Verify the output matches PyTorch reference.

### Changes Required
Add verbose comparison in test:

```python
# In test_groups_vs_pool2
print(f"Output shape: {output.shape}")
print(f"Reference shape: {ref.shape}")
print(f"Max diff: {torch.max(torch.abs(output - ref))}")
print(f"Mean diff: {torch.mean(torch.abs(output - ref))}")
```

### Test Verification
- [ ] Output shape matches reference
- [ ] Max diff is within tolerance (< 0.1 for bfloat16)
- [ ] Test PASSES

### Key Takeaways (fill after completion)
```
Step 10 Takeaways:
-
```

---

## Step 11: Expand Test Coverage

### Objective
Test with more configurations to ensure robustness.

### Changes Required
Add more test cases:

```python
@pytest.mark.parametrize(
    "batch, input_channels, output_channels, input_height, input_width, groups, kernel, stride, padding, dilation, shard_layout, dtype, weights_dtype, bias_dtype, activation, enable_act_double_buffer, enable_weight_double_buffer",
    (
        # 64 channels, 2 cores
        (1, 64, 64, 8, 8, 64, (3, 3), (1, 1), (1, 1), (1, 1), ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.bfloat16, ttnn.bfloat16, ttnn.bfloat16, None, False, False),
        # 128 channels, 4 cores
        (1, 128, 128, 8, 8, 128, (3, 3), (1, 1), (1, 1), (1, 1), ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.bfloat16, ttnn.bfloat16, ttnn.bfloat16, None, False, False),
        # With activation
        (1, 64, 64, 8, 8, 64, (3, 3), (1, 1), (1, 1), (1, 1), ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.bfloat16, ttnn.bfloat16, ttnn.bfloat16, "relu", False, False),
    ),
)
```

### Test Verification
- [ ] All test cases pass
- [ ] Different channel counts work
- [ ] Activation functions work

### Key Takeaways (fill after completion)
```
Step 11 Takeaways:
-
```

---

## Completed Steps Log

### Step 1: [COMPLETED]
```
Status: PASSED
Takeaways:
- HEIGHT_SHARDED baseline works with PCC = 0.9998038038213132 (threshold 0.99)
- Uses 2 cores with grid={(0,0)-(1,0)}, shard shape {60, 64} input / {32, 64} output
- 2D depthwise layout conversion confirmed for groups=64, channels=64, kernel=3x3
```

### Step 2: [COMPLETED]
```
Status: FAILED (as expected) - PCC = 0.0895812379007859
Takeaways:
- Test runs to completion but produces incorrect numerical results
- Root cause: Weight preparation designed for HEIGHT_SHARDED (multicast same weights to all cores)
- WIDTH_SHARDED needs per-core weight slices (each core has unique channels 32 ch/core)
- Existing function "convert_conv_weight_tensor_to_2d_depthwise_layout" is not WIDTH_SHARDED aware
```

### Step 3: [COMPLETED]
```
Status: PASSED - WIDTH_SHARDED detection works
Takeaways:
- Detection added at line 739-745 in conv2d_op_depthwise_program_factory.cpp
- Log confirms: "Depthwise conv2d sharding mode: height=false, block=false, width=true"
- TT_FATAL checkpoint triggers correctly
```

### Step 4: [COMPLETED]
```
Status: PASSED - Build succeeded, skeleton in place
Takeaways:
- Skeleton function at line 1242-1252 in prepare_conv2d_weights.cpp
- Declaration at line 83-87 in prepare_conv2d_weights.hpp
- Currently wraps existing function (placeholder)
```

### Step 5: [COMPLETED]
```
Status: PASSED - WIDTH_SHARDED wiring works
Takeaways:
- Wiring added at lines 1869-1883 in prepare_conv2d_weights.cpp
- Detects WIDTH_SHARDED via params.input_parallel_config->shard_scheme
- Gets num_cores (2) from params.input_parallel_config->grid.num_cores()
- Log confirms: "Using WIDTH_SHARDED depthwise weight prep with 2 cores"
- Still fails at program factory Step 3 checkpoint as expected
```

### Step 6: [NOT STARTED]
```
Status:
Takeaways:
```

### Step 7: [NOT STARTED] - Reader Kernel Per-Core Weight Reading
```
Status:
Takeaways:
```

### Step 8: [NOT STARTED]
```
Status:
Takeaways:
```

### Step 9: [NOT STARTED]
```
Status:
Takeaways:
```

### Step 10: [NOT STARTED]
```
Status:
Takeaways:
```

### Step 11: [NOT STARTED]
```
Status:
Takeaways:
```

---

## Debugging Tips

### 1. Use Integer Weights for Easy Debugging

In `test_groups_vs_pool2`, there's a commented section that sets weights to integer values (stick_id per kernel position). Uncomment this for easier debugging:

```python
# In test_groups_vs_pool2 (~line 5303-5308)
# UNCOMMENT THIS FOR DEBUGGING:
for out_ch in range(conv_weight_shape[0]):
    for in_ch in range(conv_weight_shape[1]):
        for kh in range(conv_weight_shape[2]):
            for kw in range(conv_weight_shape[3]):
                stick_id = kh * kernel[1] + kw + 1  # +1 to avoid zero values
                torch_weight_tensor[out_ch, in_ch, kh, kw] = stick_id
```

This makes weights predictable:
- Stick 0 (kh=0, kw=0): value = 1
- Stick 1 (kh=0, kw=1): value = 2
- Stick 2 (kh=0, kw=2): value = 3
- ...
- Stick 8 (kh=2, kw=2): value = 9

### 2. Print Raw Memory in Weight Preparation

Use the `print_raw_memory` helper function in `prepare_conv2d_weights.cpp` to dump weight tensor memory:

```cpp
// Add this helper function or use existing one
template <typename T>
void print_raw_memory(const std::vector<T>& buffer, uint32_t width, uint32_t height, const std::string& name) {
    log_info(tt::LogOp, "=== {} ({}x{}) ===", name, height, width);
    for (uint32_t row = 0; row < height; row++) {
        std::string row_str = "";
        for (uint32_t col = 0; col < width; col++) {
            uint32_t idx = row * width + col;
            if (idx < buffer.size()) {
                row_str += fmt::format("{:6.2f} ", static_cast<float>(buffer[idx]));
            }
        }
        log_info(tt::LogOp, "Row {:2d}: {}", row, row_str);
    }
}

// Call it in weight layout function:
print_raw_memory(output_buffer, padded_channels_per_shard, padded_kernel_positions, "Weight Shard 0");
```

### 3. Verify Face-by-Face Layout Visually

With integer weights, you should see this pattern in a 32x32 tile (for 32 channels, 3x3 kernel):

```
Face 0 (rows 0-15, cols 0-15):     Face 1 (rows 0-15, cols 16-31):
Row 0: 1 1 1 1 1 1 1 1 1 1 1 1...  Row 0: 1 1 1 1 1 1 1 1 1 1 1 1...
Row 1: 2 2 2 2 2 2 2 2 2 2 2 2...  Row 1: 2 2 2 2 2 2 2 2 2 2 2 2...
...                                ...
Row 8: 9 9 9 9 9 9 9 9 9 9 9 9...  Row 8: 9 9 9 9 9 9 9 9 9 9 9 9...
Row 9: 0 0 0 0 0 0 0 0 0 0 0 0...  (padding)
...
```

Each row should have the same value (stick_id) across all channel columns.

---

## Quick Reference

### File Locations
- Test: `tests/ttnn/nightly/unit_tests/operations/conv/test_conv2d.py`
- Weight prep: `ttnn/cpp/ttnn/operations/conv/conv2d/prepare_conv2d_weights.cpp`
- Program factory: `ttnn/cpp/ttnn/operations/conv/conv2d/device/conv2d_op_depthwise_program_factory.cpp`
- Reader kernel: `ttnn/cpp/ttnn/operations/pool/generic/device/kernels/dataflow/reader_pool_2d.cpp`

### Commands
```bash
# Build
./build_metal.sh --release

# Test
source python_env/bin/activate && \
TT_METAL_CLEAR_L1=0 \
TT_METAL_DPRINT_ONE_FILE_PER_RISC=1 \
TT_METAL_DPRINT_CORES="(0,0)" \
timeout 15 pytest tests/ttnn/nightly/unit_tests/operations/conv/test_conv2d.py::test_groups_vs_pool2 -v

# Recovery
tt-smi -r 1
```
