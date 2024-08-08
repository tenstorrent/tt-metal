// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/moreh_embedding/device/moreh_embedding_device_operation.hpp"
#include "ttnn/run_operation.hpp"

namespace ttnn {

namespace operations {

namespace moreh_embedding {

struct MorehEmbeddingOperation {
    static inline Tensor operator()(
        uint8_t queue_id,
        const Tensor& input,
        const Tensor& weight,
        std::optional<float> max_norm = std::nullopt,
        float norm_type = 2.0,
        std::optional<Tensor> output = std::nullopt,
        const std::optional<const DataType> dtype = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt) {
        auto device = input.device();

        auto kernel_config_val =
            init_device_compute_kernel_config(device->arch(), compute_kernel_config, MathFidelity::HiFi4);

        auto embeddings = operation::run(
                              MorehEmbeddings{
                                  .output_mem_config = memory_config.value_or(input.memory_config()),
                                  .output_dtype = dtype.value_or(weight.get_dtype()),
                                  .max_norm = max_norm,
                                  .norm_type = norm_type,
                                  .core_range = std::nullopt,
                                  .compute_kernel_config = kernel_config_val,
                              },
                              {input, weight},
                              {},
                              {output})
                              .at(0);
        return embeddings;
    }

    static inline auto operator()(
        const Tensor& input,
        const Tensor& weight,
        std::optional<bool> max_norm = std::nullopt,
        float norm_type = 2.0,
        std::optional<Tensor> output = std::nullopt,
        const std::optional<const DataType> dtype = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt) {
        constexpr auto DefaultQueueId = 0;
        return operator()(
            DefaultQueueId, input, weight, max_norm, norm_type, output, dtype, memory_config, compute_kernel_config);
    }
};

}  // namespace moreh_embedding
}  // namespace operations

constexpr auto moreh_embedding = ttnn::register_operation_with_auto_launch_op<
    "ttnn::moreh_embedding",
    ttnn::operations::moreh_embedding::MorehEmbeddingOperation>();

}  // namespace ttnn
