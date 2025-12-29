# Depthwise Conv2D Width Sharded Implementation Guide

This document provides a detailed implementation guide for adding width sharded support to depthwise conv2d while preserving the face-by-face weight layout.

## Table of Contents
1. [Current State Analysis](#current-state-analysis)
2. [Width Sharded Requirements](#width-sharded-requirements)
3. [Weight Layout Strategy](#weight-layout-strategy)
4. [Implementation Steps](#implementation-steps)
5. [Code Changes Required](#code-changes-required)
6. [Testing Strategy](#testing-strategy)

---

## Current State Analysis

### Current Height Sharded Implementation

**Weight Layout** (`prepare_conv2d_weights.cpp:1088-1186`):
```
Input:  [out_channels, 1, kernel_h, kernel_w]
Output: [padded_channels, padded_kernel_positions, 1, 1]
```

#### Face-by-Face Filling Pattern (Height Sharded)

**KEY PRINCIPLE**: Fill each face completely until it runs out of space, then move to the next face. If weights span multiple tiles, continue filling the second tile in the same sequential manner.

For 32 channels with 3x3 kernel (9 kernel positions, 9 sticks):

```
Single 32x32 Tile Layout:
┌─────────────────┬─────────────────┐
│ Face 0          │ Face 1          │  rows 0-15
│ cols 0-15       │ cols 16-31      │
│                 │                 │
│ Stick 0: ch0-15 │ Stick 8: ch0-15 │  row 0
│ Stick 1: ch0-15 │ Stick 8: ch16-31│  row 1
│ Stick 2: ch0-15 │ (padding)       │  row 2
│ Stick 3: ch0-15 │ (padding)       │  row 3
│ Stick 4: ch0-15 │ ...             │  row 4
│ Stick 5: ch0-15 │                 │  row 5
│ Stick 6: ch0-15 │                 │  row 6
│ Stick 7: ch0-15 │                 │  row 7
│ (padding)       │                 │  rows 8-15
├─────────────────┼─────────────────┤
│ Face 2          │ Face 3          │  rows 16-31
│ cols 0-15       │ cols 16-31      │
│                 │                 │
│ Stick 0: ch16-31│ Stick 8: ch16-31│  row 16
│ Stick 1: ch16-31│ (padding)       │  row 17
│ Stick 2: ch16-31│ ...             │  ...
│ ...             │                 │
│ Stick 7: ch16-31│                 │
│ (padding)       │                 │
└─────────────────┴─────────────────┘

Filling order:
1. Fill Face 0 completely (stick by stick, until face is full or no more sticks)
2. Move to Face 1, continue filling
3. Move to Face 2, continue filling
4. Move to Face 3, continue filling
5. If more data remains, move to Tile 1 and repeat
```

#### Multi-Tile Continuation (Larger Weight Sizes)

**IMPORTANT**: When weight size exceeds one tile, we continue filling the second tile in the SAME sequential face-by-face manner:

```
Example: 64 channels, 3x3 kernel - weights span multiple tiles

Filling sequence across tiles:
┌─────────────────────────────────────────────────────────────────┐
│ TILE 0                              │ TILE 1                    │
│ ┌───────────┬───────────┐           │ ┌───────────┬───────────┐ │
│ │ Face 0    │ Face 1    │           │ │ Face 0    │ Face 1    │ │
│ │ (fill     │ (fill     │           │ │ (continue │ (continue │ │
│ │  first)   │  second)  │           │ │  filling) │  filling) │ │
│ ├───────────┼───────────┤           │ ├───────────┼───────────┤ │
│ │ Face 2    │ Face 3    │           │ │ Face 2    │ Face 3    │ │
│ │ (fill     │ (fill     │           │ │ (continue │ (continue │ │
│ │  third)   │  fourth)  │           │ │  filling) │  filling) │ │
│ └───────────┴───────────┘           │ └───────────┴───────────┘ │
└─────────────────────────────────────────────────────────────────┘

Order: Tile0.Face0 → Tile0.Face1 → Tile0.Face2 → Tile0.Face3 →
       Tile1.Face0 → Tile1.Face1 → Tile1.Face2 → Tile1.Face3 → ...

Each face is filled completely before moving to the next face.
When a tile is full, continue to the next tile with the same pattern.
```

#### Padding Before Tilization (Critical!)

**WHY PADDING IS REQUIRED**: The face-by-face filling requires the output buffer to be padded to tile boundaries BEFORE the data is placed. This is because:

1. **Buffer size constraint**: The tilization hardware expects data aligned to 32x32 tile boundaries
2. **Face alignment**: Each face is 16x16, and we fill faces sequentially
3. **Row padding**: When channels < 32 (e.g., 16 channels), we need padding rows between sticks so each stick ends up in its own tile row after tilization

```
Example: 16 channels, 3x3 kernel

Without padding (WRONG - buffer too small):
┌─────────────────┐
│ Stick0: 16 vals │  ← Only 16 values, but face is 16 wide
│ Stick1: 16 vals │  ← Next stick immediately follows
│ ...             │
└─────────────────┘
Problem: After tilization, sticks are not properly aligned to tile rows

With padding (CORRECT):
┌─────────────────┬─────────────────┐
│ Face 0          │ Face 1          │
│ Stick0: ch0-15  │ (zeros ch16-31) │  row 0: data + padding
│ (zeros)         │ (zeros)         │  row 1: padding row
│ Stick1: ch0-15  │ (zeros ch16-31) │  row 2: data + padding
│ (zeros)         │ (zeros)         │  row 3: padding row
│ ...             │ ...             │
└─────────────────┴─────────────────┘

Padding ensures:
- Channels padded to 32 (tile width)
- Kernel positions padded to 32 (tile height)
- Each stick occupies ceil(padded_channels/16) rows with proper spacing
```

#### Code Reference for Padding Calculation

```cpp
// From prepare_conv2d_weights.cpp:1104-1113
constexpr uint32_t TILE_SIZE = 32;
constexpr uint32_t FACE_SIZE = 16;

// Pad channels to tile size for proper tilization
uint32_t padded_out_channels = ((out_channels + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE;

// Calculate rows per stick (for face-by-face placement)
uint32_t data_rows_per_stick = (out_channels + FACE_SIZE - 1) / FACE_SIZE;        // Actual data rows
uint32_t rows_per_stick_padded = (padded_out_channels + FACE_SIZE - 1) / FACE_SIZE; // Padded rows

// Pad kernel positions to tile size
uint32_t padded_kernel_positions = ((total_kernel_positions + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE;

// Output buffer must be large enough for padded dimensions
ttnn::Shape output_shape{padded_out_channels, padded_kernel_positions, 1, 1};
```

**Weight Distribution** (`conv2d_op_depthwise_program_factory.cpp:593-678`):
- Single sender (Core 0) reads ALL weights from DRAM
- Multicasts to all receiver cores
- Works because all cores process all channels (different NHW positions)

### Width Sharded Grid Layout

```
Width Sharded (4 cores, 64 channels):
┌─────────┬─────────┬─────────┬─────────┐
│ Core 0  │ Core 1  │ Core 2  │ Core 3  │
│ NHW:ALL │ NHW:ALL │ NHW:ALL │ NHW:ALL │
│ Ch:0-15 │ Ch:16-31│ Ch:32-47│ Ch:48-63│
└─────────┴─────────┴─────────┴─────────┘

Each core:
- Processes ALL spatial positions (batch * height * width)
- Processes SUBSET of channels (channels_per_core = total / num_cores)
- Needs ONLY weights for its channel subset
```

---

## Width Sharded Requirements

### Key Insight
**NO MULTICAST NEEDED** - Each core has unique channels, no weight sharing.

### What Each Core Needs
| Core | Channel Range | Weight Offset | Weight Size |
|------|--------------|---------------|-------------|
| 0    | [0, C/N)     | 0             | W_size/N    |
| 1    | [C/N, 2C/N)  | W_size/N      | W_size/N    |
| ...  | ...          | ...           | ...         |
| N-1  | [(N-1)C/N, C)| (N-1)*W_size/N| W_size/N    |

Where:
- C = total channels
- N = number of cores
- W_size = total weight size in bytes

---

## Weight Layout Strategy

### Option A: Per-Shard Weight Preparation (Recommended)

Create weights where each shard is a complete, independent unit:

```
Original weights: [64, 1, 3, 3] (64 channels, 3x3 kernel)
Prepared weights for 4 cores:

Shard 0 (channels 0-15):  [32, 32, 1, 1] face-by-face layout
Shard 1 (channels 16-31): [32, 32, 1, 1] face-by-face layout
Shard 2 (channels 32-47): [32, 32, 1, 1] face-by-face layout
Shard 3 (channels 48-63): [32, 32, 1, 1] face-by-face layout

Memory layout: [Shard0 | Shard1 | Shard2 | Shard3]
```

**Advantages**:
- Each core reads exactly one tile (or contiguous tiles)
- Preserves face-by-face layout within each shard
- Simple offset calculation: `core_idx * shard_size_bytes`

### Option B: Global Layout with Strided Access

Keep existing global layout, each core calculates its offset:

```
Global weights: [padded_channels, padded_kernel_positions, 1, 1]

Core 0 reads: columns 0-15 (channels 0-15)
Core 1 reads: columns 16-31 (channels 16-31)
...
```

**Disadvantage**: Complex strided reads, not contiguous

---

## Implementation Steps

### Step 1: Add Width Sharded Detection

**File**: `conv2d_op_depthwise_program_factory.cpp`

```cpp
// In multi_core_conv2d_depthwise_impl(), after line ~318
bool is_height_sharded = (a.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED);
bool is_block_sharded = (a.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED);
bool is_width_sharded = (a.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED);

if (is_width_sharded) {
    // Width sharded path - no multicast, direct DRAM read per core
    // ... (implementation below)
} else {
    // Existing height sharded path with multicast
    // ... (current code)
}
```

### Step 2: Create Width Sharded Weight Preparation Function

**File**: `prepare_conv2d_weights.cpp`

```cpp
/*
Converts 2D depthwise convolution weights to face-by-face layout for width sharded conv.
Creates separate shards for each core, preserving face-by-face layout within each shard.

Parameters:
- conv_weight_tensor: Input weights [out_channels, 1, kernel_h, kernel_w]
- num_channel_shards: Number of cores (channel shards)
- output_dtype: Output data type

Returns:
- Tensor with shape [num_shards * padded_channels_per_shard, padded_kernel_positions, 1, 1]
  where each shard is independently laid out face-by-face
*/
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

    // Calculate per-shard dimensions
    uint32_t channels_per_shard = out_channels / num_channel_shards;

    constexpr uint32_t TILE_SIZE = 32;
    constexpr uint32_t FACE_SIZE = 16;

    // Pad per-shard channels to tile size
    uint32_t padded_channels_per_shard = ((channels_per_shard + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE;
    uint32_t padded_kernel_positions = ((total_kernel_positions + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE;

    // Output shape: all shards concatenated
    uint32_t total_padded_channels = padded_channels_per_shard * num_channel_shards;
    ttnn::Shape output_shape{total_padded_channels, padded_kernel_positions, 1, 1};

    auto compute = [&](const tt::tt_metal::HostBuffer& input_host_buffer) {
        auto input_buffer = tt::tt_metal::host_buffer::get_as<T>(input_host_buffer);
        auto output_buffer = std::vector<T>(output_shape.volume(), static_cast<T>(0));

        uint32_t data_rows_per_stick = (channels_per_shard + FACE_SIZE - 1) / FACE_SIZE;
        uint32_t rows_per_stick_padded = (padded_channels_per_shard + FACE_SIZE - 1) / FACE_SIZE;
        uint32_t shard_size = padded_channels_per_shard * padded_kernel_positions;

        // Process each shard independently
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

                        // Get input value from [ch, 0, kh, kw]
                        auto input_idx = tt::tt_metal::compute_flat_indices(
                            ttnn::SmallVector<int>{(int)global_ch, 0, (int)kh, (int)kw},
                            compute_strides(original_weight_shape));
                        T value = input_buffer[input_idx];

                        // Place in shard's face-by-face layout
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
            // ... other types
        };

    output_dtype = ((output_dtype == DataType::BFLOAT8_B) || (output_dtype == DataType::BFLOAT4_B))
                       ? DataType::FLOAT32 : output_dtype;

    return layout_map.at(conv_weight_tensor.dtype())(
        conv_weight_tensor, original_shape, num_channel_shards, output_dtype);
}
```

### Step 3: Modify Program Factory for Width Sharded

**File**: `conv2d_op_depthwise_program_factory.cpp`

Replace the weight multicast setup section (~lines 593-678) with:

```cpp
// ============================================================
// Weight Distribution Setup
// ============================================================

bool is_width_sharded = (a.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED);

if (is_width_sharded) {
    // ============================================================
    // WIDTH SHARDED: Each core reads its own weights directly from DRAM
    // No multicast needed - each core has unique channels
    // ============================================================

    uint32_t channels_per_core = output_channels / num_cores;
    uint32_t padded_channels_per_core = tt::round_up(channels_per_core, 32);
    uint32_t padded_kernel_positions = tt::round_up(kernel_h * kernel_w, 32);

    // Calculate shard size in bytes (one tile = 32x32 elements)
    uint32_t shard_ntiles = (padded_channels_per_core * padded_kernel_positions) / (32 * 32);
    uint32_t weight_tile_nbytes = tt::tile_size(params.data_format);
    uint32_t shard_size_bytes = shard_ntiles * weight_tile_nbytes;

    log_debug(tt::LogOp, "Width sharded weight distribution:");
    log_debug(tt::LogOp, "  channels_per_core: {}", channels_per_core);
    log_debug(tt::LogOp, "  shard_ntiles: {}", shard_ntiles);
    log_debug(tt::LogOp, "  shard_size_bytes: {}", shard_size_bytes);

    // Set up runtime args for each core - each reads its own weight slice
    for (uint32_t core_idx = 0; core_idx < num_cores; core_idx++) {
        CoreCoord core = all_cores[core_idx];

        // Calculate weight offset for this core's channel shard
        uint32_t weight_offset = core_idx * shard_size_bytes;

        std::vector<uint32_t> reader_args = {
            static_cast<uint32_t>(weight_buffer_addr + weight_offset),  // weight_addr_dram_base
            1,  // is_sender = true (each core reads for itself, no multicast)
            0, 0, 0, 0,  // mcast coords (unused)
            0,  // num_dests = 0 (no multicast)
            0,  // num_mcast_cores = 0
            0,  // sender_semaphore (unused)
            0   // receiver_semaphore (unused)
        };

        SetRuntimeArgs(program, reader0_kernel, core, reader_args);
    }

    log_debug(tt::LogOp, "Width sharded: {} cores, each reading {} bytes from DRAM",
              num_cores, shard_size_bytes);

} else {
    // ============================================================
    // HEIGHT SHARDED (existing code): Multicast from first core
    // ============================================================

    // ... existing multicast code ...
}
```

### Step 4: Modify Reader Kernel for Width Sharded

**File**: `reader_pool_2d.cpp`

The current kernel already handles the case where `weights_mcast_num_dests == 0` (no multicast).
Just ensure it reads from the correct offset provided in runtime args.

Key changes needed at line ~507-627:

```cpp
if constexpr (reader_id == 0) {
    // ... existing setup ...

    if (is_sender) {
        // Get runtime args
        uint32_t weight_addr_with_offset = get_arg_val<uint32_t>(0);  // Already includes core's offset
        uint32_t weights_mcast_num_dests = get_arg_val<uint32_t>(6);

        // Read weights for this core from DRAM
        cb_reserve_back(weight_cb_id, weight_ntiles);
        uint32_t weight_l1_addr = get_write_ptr(weight_cb_id);
        uint32_t weights_start_address = weight_l1_addr;

        for (uint32_t tile_id = 0; tile_id < weight_ntiles; tile_id++) {
            noc_async_read_tile(tile_id, s_weight, weight_l1_addr);
            weight_l1_addr += weight_tile_nbytes;
        }
        noc_async_read_barrier();

        // For width sharded (num_dests == 0), skip multicast
        if (weights_mcast_num_dests > 0) {
            // ... existing multicast code ...
        }

        cb_push_back(weight_cb_id, weight_ntiles);
    }
    // ... receiver path (only used for height sharded) ...
}
```

### Step 5: Update Weight Preparation Call

**File**: `prepare_conv2d_weights.cpp` at line ~1853

```cpp
// In prepare_conv_weights() function
if (is_depthwise) {
    TensorMemoryLayout shard_layout = conv_config.shard_layout.value_or(TensorMemoryLayout::HEIGHT_SHARDED);

    if (shard_layout == TensorMemoryLayout::WIDTH_SHARDED) {
        uint32_t num_cores = parallel_config.grid.num_cores();
        weight_tensor_ = convert_conv_weight_tensor_to_2d_depthwise_layout_width_sharded(
            weight_tensor_, num_cores, weight_tensor_.dtype());
    } else {
        // Height sharded - existing code
        weight_tensor_ = convert_conv_weight_tensor_to_2d_depthwise_layout(
            weight_tensor_, weight_tensor_.dtype());
    }
}
```

---

## Code Changes Required

### Summary of Files to Modify

| File | Changes |
|------|---------|
| `prepare_conv2d_weights.cpp` | Add `convert_conv_weight_tensor_to_2d_depthwise_layout_width_sharded()` function |
| `prepare_conv2d_weights.hpp` | Add function declaration |
| `conv2d_op_depthwise_program_factory.cpp` | Add width sharded detection and per-core weight offset setup |

### Estimated Lines of Code

- Weight preparation function: ~100 lines
- Program factory changes: ~50 lines
- Header declaration: ~5 lines

**Total: ~155 lines of new/modified code**

---

## Testing Strategy

### Unit Tests

```python
# test_conv2d.py

@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("channels", [32, 64, 128])
@pytest.mark.parametrize("kernel_size", [(3, 3), (5, 5)])
@pytest.mark.parametrize("num_cores", [1, 2, 4, 8])
def test_depthwise_conv2d_width_sharded(batch_size, channels, kernel_size, num_cores):
    # Ensure channels divisible by num_cores
    if channels % num_cores != 0:
        pytest.skip("Channels must be divisible by num_cores")

    input_shape = (batch_size, channels, 32, 32)
    # ... create input tensor with WIDTH_SHARDED memory config ...
    # ... run depthwise conv2d ...
    # ... compare with torch reference ...
```

### Edge Cases to Test

1. **Channel alignment**: channels = 16, 32, 48, 64 (various tile alignments)
2. **Core count variations**: 1, 2, 4, 8 cores
3. **Kernel sizes**: 1x1, 3x3, 5x5, 7x7
4. **Non-square kernels**: 1x3, 3x1, 1x5

### Validation Approach

1. Run depthwise conv2d with WIDTH_SHARDED config
2. Compare output against:
   - PyTorch `nn.Conv2d(groups=channels)` reference
   - HEIGHT_SHARDED version (should produce identical results)
3. Verify each core processes correct channel subset (debug logging)

---

## Summary

The width sharded implementation for depthwise conv2d is simpler than block sharded because:

1. **No multicast needed** - each core reads its own unique channel slice
2. **Preserve face-by-face layout** - just apply it per-shard instead of globally
3. **Simple offset calculation** - `core_idx * shard_size_bytes`

The main work is:
1. New weight preparation function that creates per-shard face-by-face layouts
2. Program factory changes to detect width sharded and set per-core weight offsets
3. Ensure reader kernel handles `num_dests == 0` case (already does)
