// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"

#include "groupnorm_v3_device_operation_types.hpp"
#include "groupnorm_v3_program_factory.hpp"

namespace ttnn::operations::normalization::group_norm_v3 {

struct GroupNormV3DeviceOperation {
    using operation_attributes_t = group_norm_v3::operation_attributes_t;
    using tensor_args_t = group_norm_v3::tensor_args_t;
    using spec_return_value_t = group_norm_v3::spec_return_value_t;
    using tensor_return_value_t = group_norm_v3::tensor_return_value_t;
    using program_factory_t = std::variant<program::GroupNormV3ProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t& args, const tensor_args_t&);
};

}  // namespace ttnn::operations::normalization::group_norm_v3

namespace ttnn::prim {

// Direct function in ttnn::prim namespace (aligned with Artem's PR #35013, #35015)
// No register_operation - uses ttnn::device_operation::launch<> directly
ttnn::operations::normalization::group_norm_v3::tensor_return_value_t group_norm_v3(
    const Tensor& input,
    uint32_t num_groups,
    float eps,
    tt::tt_metal::DataType output_dtype,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const CoreCoord& core_grid,
    bool inplace,
    int chunk_size,
    const DeviceComputeKernelConfig& compute_kernel_config,
    std::optional<Tensor> gamma = std::nullopt,
    std::optional<Tensor> beta = std::nullopt);

}  // namespace ttnn::prim
