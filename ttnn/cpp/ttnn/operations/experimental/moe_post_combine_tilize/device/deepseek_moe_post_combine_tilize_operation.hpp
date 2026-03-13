// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "deepseek_moe_post_combine_tilize_device_operation_types.hpp"
#include "deepseek_moe_post_combine_tilize_program_factory.hpp"

#include "ttnn/decorators.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct DeepseekMoEPostCombineTilizeDeviceOperation {
    using operation_attributes_t = DeepseekMoEPostCombineTilizeParams;
    using tensor_args_t = DeepseekMoEPostCombineTilizeInputs;
    using spec_return_value_t = ttnn::TensorSpec;
    using tensor_return_value_t = std::vector<ttnn::Tensor>;
    using program_factory_t = std::variant<DeepseekMoEPostCombineTilizeProgramFactory>;
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::vector<ttnn::Tensor> deepseek_moe_post_combine_tilize(
    const ttnn::Tensor& input_tensor,
    uint32_t dim,
    uint64_t split_size,
    const tt::tt_metal::MemoryConfig& output_memory_config,
    const ttnn::DeviceComputeKernelConfig& compute_kernel_config);

}  // namespace ttnn::prim
