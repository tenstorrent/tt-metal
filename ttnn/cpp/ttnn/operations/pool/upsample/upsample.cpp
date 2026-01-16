// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "upsample.hpp"
#include <algorithm>
#include "tt-metalium/shape.hpp"
#include "ttnn/operations/pool/upsample/device/upsample_device_operation.hpp"
#include "tt-metalium/buffer_types.hpp"
#include "tt-metalium/work_split.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"
#include "ttnn/operations/sliding_window/halo/halo.hpp"

namespace ttnn::operations::upsample {

static std::pair<Tensor, sliding_window::SlidingWindowConfig> apply_bilinear_halo_preprocessing(
    const ttnn::Tensor& input_tensor, int scale_h, int scale_w) {
    // Create sliding window config for bilinear upsample (fixed 2x2 kernel, 1x1 stride, 1x1x1x1 padding)
    const tt::tt_metal::Shape& input_shape = input_tensor.logical_shape();
    const uint32_t batch_size = input_tensor.padded_shape()[0];
    const uint32_t input_height = input_tensor.padded_shape()[1];
    const uint32_t input_width = input_tensor.padded_shape()[2];
    const uint32_t channels = input_shape[3];
    const uint32_t num_cores_nhw = input_tensor.shard_spec().value().num_cores();
    const uint32_t num_cores_c = 1;

    sliding_window::SlidingWindowConfig sliding_window_config = sliding_window::SlidingWindowConfig{
        .batch_size = batch_size,
        .channels = channels,  // IMPORTANT: Add channels for program factory to access
        .input_hw = {input_height, input_width},
        .window_hw = {2, 2},        // kernel size
        .stride_hw = {1, 1},        // stride
        .padding = {{1, 1, 1, 1}},  // padding (all sides)
        .dilation_hw = {1, 1},      // dilation
        .scale_h = scale_h,         // upsampling scale factor height
        .scale_w = scale_w,         // upsampling scale factor width
        .num_cores_nhw = num_cores_nhw,
        .num_cores_c = num_cores_c,
        .core_range_set = input_tensor.memory_config().shard_spec().value().grid,
        .snap_to_tile = false,
        .is_bilinear = true};

    // Reshape input tensor to {1, 1, N*H*W, C}
    const tt::tt_metal::Shape new_shape({1, 1, input_shape[0] * input_shape[1] * input_shape[2], input_shape[3]});
    ttnn::Tensor input_tensor_reshaped = ttnn::reshape(input_tensor, new_shape);

    tt::tt_metal::Tensor haloed_tensor = ttnn::halo(
        input_tensor_reshaped,
        sliding_window_config,
        0,      // pad_val
        false,  // remote_read
        false,  // transpose_mcast
        input_tensor_reshaped.memory_config(),
        false);  // is_out_tiled

    return {haloed_tensor, sliding_window_config};
}

static tt::tt_metal::MemoryConfig compute_bilinear_autoshard_memory_config(const ttnn::Tensor& input_tensor) {
    const tt::tt_metal::Shape& input_shape = input_tensor.logical_shape();
    const tt::tt_metal::CoreCoord compute_grid_size = input_tensor.device()->compute_with_storage_grid_size();

    const uint32_t total_input_sticks = input_shape[0] * input_shape[1] * input_shape[2];
    const uint32_t max_num_cores = compute_grid_size.x * compute_grid_size.y;
    const uint32_t num_shards = std::min(max_num_cores, total_input_sticks);
    const uint32_t shard_height = tt::round_up(total_input_sticks, num_shards) / num_shards;

    const tt::tt_metal::ShardSpec shard_spec = tt::tt_metal::ShardSpec(
        tt::tt_metal::num_cores_to_corerangeset(num_shards, compute_grid_size, true),
        {shard_height, input_shape[3]},
        tt::tt_metal::ShardOrientation::ROW_MAJOR);

    return tt::tt_metal::MemoryConfig(
        tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED, tt::tt_metal::BufferType::L1, shard_spec);
}

ttnn::Tensor ExecuteUpSample::invoke(
    const ttnn::Tensor& input_tensor,
    std::variant<int, std::array<int, 2>, float, std::array<float, 2>> scale_factor,
    const std::string& mode,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    tt::tt_metal::MemoryConfig mem_config = output_mem_config.value_or(input_tensor.memory_config());

    // Parse scale factors from variant - extract as floats
    float scale_h = 1.0f;
    float scale_w = 1.0f;

    std::visit(
        [&scale_h, &scale_w](auto&& sf) {
            using T = std::decay_t<decltype(sf)>;
            if constexpr (std::is_same_v<T, int>) {
                scale_h = static_cast<float>(sf);
                scale_w = static_cast<float>(sf);
            } else if constexpr (std::is_same_v<T, std::array<int, 2>>) {
                scale_h = static_cast<float>(sf[0]);
                scale_w = static_cast<float>(sf[1]);
            } else if constexpr (std::is_same_v<T, float>) {
                scale_h = sf;
                scale_w = sf;
            } else if constexpr (std::is_same_v<T, std::array<float, 2>>) {
                scale_h = sf[0];
                scale_w = sf[1];
            } else {
                static_assert(sizeof(T) != 0, "Type check failed.");
            }
        },
        scale_factor);

    // Validation is handled by the device operation's validate_on_program_cache_miss

    ttnn::DeviceComputeKernelConfig config = compute_kernel_config.value_or(
        ttnn::init_device_compute_kernel_config(input_tensor.device()->arch(), std::nullopt, MathFidelity::HiFi4));

    // For bilinear mode, call halo preprocessing step before the upsample operation
    if (mode == "bilinear") {
        // Autoshard if needed
        ttnn::Tensor input_for_halo = input_tensor;
        tt::tt_metal::MemoryConfig output_mem_config_to_use = mem_config;

        if (!input_tensor.is_sharded()) {
            // Bilinear mode with non sharded input is not supported. Performing autosharding
            tt::tt_metal::MemoryConfig sharded_memory_config = compute_bilinear_autoshard_memory_config(input_for_halo);
            input_for_halo = to_memory_config(input_for_halo, sharded_memory_config);
            output_mem_config_to_use = sharded_memory_config;
        }

        // Apply halo preprocessing (requires integer scales)
        int scale_h_int = static_cast<int>(scale_h);
        int scale_w_int = static_cast<int>(scale_w);
        std::pair<tt::tt_metal::Tensor, sliding_window::SlidingWindowConfig> halo_result =
            apply_bilinear_halo_preprocessing(input_for_halo, scale_h_int, scale_w_int);
        tt::tt_metal::Tensor haloed_tensor = halo_result.first;
        sliding_window::SlidingWindowConfig sliding_window_config = halo_result.second;

        // Pass the HALOED tensor to upsample for bilinear mode
        ttnn::Tensor output_tensor = ttnn::prim::upsample(
            haloed_tensor, scale_h, scale_w, mode, output_mem_config_to_use, config, sliding_window_config);

        // Convert to final memory config if needed
        if (output_mem_config.has_value() && output_tensor.memory_config() != output_mem_config) {
            output_tensor = to_memory_config(output_tensor, output_mem_config.value());
        }
        return output_tensor;
    }

    // For nearest mode, pass to unified prim (device op handles factory selection)
    return ttnn::prim::upsample(input_tensor, scale_h, scale_w, mode, mem_config, config);
}
}  // namespace ttnn::operations::upsample
