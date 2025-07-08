// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "upsample.hpp"
#include <algorithm>
#include "device/upsample_op.hpp"
#include "tt-metalium/buffer_types.hpp"
#include "tt-metalium/work_split.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::upsample {
using namespace tt;
using namespace tt::tt_metal;

static ShardSpec compute_bilinear_autoshard_spec(const ttnn::Tensor& input_tensor) {
    /*
    Calculates the sharding spec required for bilinear transform
    */
    const auto& input_shape = input_tensor.logical_shape();
    const auto batch_size = input_shape[0];
    const auto input_h = input_shape[1];
    const auto input_w = input_shape[2];
    const auto num_channels = input_shape[3];
    auto compute_grid_size = input_tensor.device()->compute_with_storage_grid_size();
    uint32_t max_num_cores = compute_grid_size.x * compute_grid_size.y;
    uint32_t max_num_shards = batch_size * input_h * input_w;
    uint32_t num_shards = std::min(max_num_cores, max_num_shards);
    while (num_shards > 0) {
        if ((batch_size * input_h % num_shards == 0) &&
            ((batch_size * input_h * input_w / num_shards) % input_w == 0)) {
            break;
        }
        num_shards--;
    }
    uint32_t num_cores = num_shards;
    CoreRangeSet core_range_set = num_cores_to_corerangeset(num_cores, compute_grid_size, true);
    uint32_t shard_height = batch_size * input_h * input_w / num_cores;
    uint32_t shard_width = num_channels;
    auto shard_orientation = ShardOrientation::ROW_MAJOR;
    return ShardSpec(core_range_set, {shard_height, shard_width}, shard_orientation);
}

ttnn::Tensor ExecuteUpSample::invoke(
    const ttnn::Tensor& input_tensor,
    std::variant<int, tt::tt_metal::Array2D> scale_factor,
    const std::string& mode,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    MemoryConfig mem_config = output_mem_config.value_or(input_tensor.memory_config());

    int scale_h = 1;
    int scale_w = 1;
    std::visit(
        [&scale_h, &scale_w](auto&& sf) {
            using T = std::decay_t<decltype(sf)>;
            if constexpr (std::is_same_v<T, int>) {
                scale_h = sf;
                scale_w = sf;
            } else if constexpr (std::is_same_v<T, tt::tt_metal::Array2D>) {
                scale_h = sf.at(0);
                scale_w = sf.at(1);
            } else {
                // static_assert(false, "Unsupported scale factor");
                static_assert(sizeof(T) != 0, "Type check failed.");
            }
        },
        scale_factor);

    if (!input_tensor.is_sharded() && mode == "bilinear") {
        // Bilinear mode with non sharded input is not supported. Performing autosharding

        auto input_tensor_sharded = input_tensor;
        auto memory_layout = TensorMemoryLayout::HEIGHT_SHARDED;
        auto buffer_type = BufferType::L1;

        ShardSpec shard_spec = compute_bilinear_autoshard_spec(input_tensor_sharded);
        MemoryConfig sharded_memory_config = MemoryConfig(memory_layout, buffer_type, shard_spec);
        input_tensor_sharded = to_memory_config(input_tensor_sharded, sharded_memory_config);
        ttnn::DeviceComputeKernelConfig config = compute_kernel_config.value_or(ttnn::init_device_compute_kernel_config(
            input_tensor_sharded.device()->arch(), std::nullopt, MathFidelity::HiFi4));
        // Output sharding should be the same as input
        // Output shard shape gets rescaled in op
        auto output_tensor =
            tt::tt_metal::operation::run(
                UpSample{scale_h, scale_w, mode, sharded_memory_config, config}, {input_tensor_sharded})
                .front();

        if (output_mem_config.has_value() && output_tensor.memory_config() != output_mem_config) {
            output_tensor = to_memory_config(output_tensor, output_mem_config.value());
        }
        return output_tensor;
    }

    ttnn::DeviceComputeKernelConfig config = compute_kernel_config.value_or(
        ttnn::init_device_compute_kernel_config(input_tensor.device()->arch(), std::nullopt, MathFidelity::HiFi4));
    // return ttnn::upsample(input_tensor, scale_h, scale_w, mem_config);
    auto output_tensor =
        tt::tt_metal::operation::run(UpSample{scale_h, scale_w, mode, mem_config, config}, {input_tensor}).front();
    return output_tensor;
}

}  // namespace ttnn::operations::upsample
