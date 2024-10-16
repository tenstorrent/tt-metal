// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device/group_attn_matmul_device_operation.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/core/core.hpp"
#include "group_attn_matmul.hpp"

namespace ttnn::operations::experimental::matmul {

ttnn::Tensor GroupAttnMatmulOperation::invoke(
    uint8_t queue_id,
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const CoreCoord& compute_with_storage_grid_size,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<const DataType> output_dtype,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    std::optional<Tensor> optional_output_tensor) {
    auto mem_config = memory_config.value_or(input_tensor_a.memory_config());
    bool row_major = false;
    // GroupAttnMatmul::validate will check that any sharded memory configs have same orientation
    if (input_tensor_a.is_sharded()) {
        row_major = input_tensor_a.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR;
    } else if (input_tensor_b.is_sharded()) {
        row_major = input_tensor_b.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR;
    } else if (mem_config.is_sharded()) {
        if (mem_config.shard_spec.has_value()) {
            row_major = mem_config.shard_spec.value().orientation == ShardOrientation::ROW_MAJOR;
        }
    }

    auto arch = input_tensor_a.storage_type() == StorageType::DEVICE
                    ? input_tensor_a.device()->arch()
                    : ttnn::operations::experimental::auto_format::AutoFormat::GetDefaultDevice()->arch();
    auto kernel_config_val = init_device_compute_kernel_config(arch, compute_kernel_config);

    // Need to cache on out_subblock_w because it must be a compile time arg for optimal use of templated pack_untilize
    // APIs
    const uint32_t Nt = input_tensor_b.get_legacy_shape()[-1] / tt::constants::TILE_WIDTH;
    constexpr uint32_t HALF_DST_MAX = 8;  // 8 is the max number of tiles for half DST (assuming out_subblock_h == 1)
    constexpr uint32_t HALF_DST_MAX_FP32 = 4;  // max dst tiles are 4 for fp32
    uint32_t out_subblock_w;

    std::visit(
        [&](auto&& kernel_config_val) {
            using T = std::decay_t<decltype(kernel_config_val)>;
            if constexpr (std::is_same_v<T, WormholeComputeKernelConfig>) {
                out_subblock_w =
                    kernel_config_val.fp32_dest_acc_en ? std::min(Nt, HALF_DST_MAX_FP32) : std::min(Nt, HALF_DST_MAX);
            } else {
                out_subblock_w = std::min(Nt, HALF_DST_MAX);
            }
        },
        kernel_config_val);

    return operation::run(GroupAttnMatmulDeviceOperation{std::nullopt,
                                                         std::nullopt,
                                                         out_subblock_w,
                                                         compute_with_storage_grid_size,
                                                         mem_config,
                                                         output_dtype.value_or(input_tensor_a.get_dtype()),
                                                         row_major,
                                                         kernel_config_val},
                          {input_tensor_a, input_tensor_b},
                          {},
                          {optional_output_tensor},
                          queue_id)
        .at(0);
}

ttnn::Tensor GroupAttnMatmulOperation::invoke(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const CoreCoord& compute_with_storage_grid_size,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<const DataType> output_dtype,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    std::optional<Tensor> optional_output_tensor) {
    return invoke(ttnn::DefaultQueueId,
                  input_tensor_a,
                  input_tensor_b,
                  compute_with_storage_grid_size,
                  memory_config,
                  output_dtype,
                  compute_kernel_config,
                  optional_output_tensor);
}

};  // namespace ttnn::operations::experimental::matmul
