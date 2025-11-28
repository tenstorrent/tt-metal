// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "transpose_device_operation_types.hpp"
#include "transpose_program_factory.hpp"

#include "ttnn/decorators.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

#include <variant>

namespace ttnn::operations::data_movement {

struct TransposeDeviceOperation {
    using operation_attributes_t = transpose::operation_attributes_t;
    using tensor_args_t = transpose::tensor_args_t;
    using spec_return_value_t = transpose::spec_return_value_t;
    using tensor_return_value_t = transpose::tensor_return_value_t;
    using program_factory_t = std::variant<
        program::TransposeWHProgramFactory,
        program::TransposeWHShardedProgramFactory,
        program::TransposeWHShardedRMProgramFactory,
        program::TransposeHCProgramFactory,
        program::TransposeHCShardedProgramFactory,
        program::TransposeCNProgramFactory>;

    static program_factory_t select_program_factory(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_hit(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input_tensor,
        TransposeOpDim dim,
        const tt::tt_metal::MemoryConfig& output_mem_config,
        const std::optional<float>& pad_value);
};

}  // namespace ttnn::operations::data_movement

namespace ttnn::prim {
constexpr auto transpose =
    ttnn::register_operation<"ttnn::prim::transpose", ttnn::operations::data_movement::TransposeDeviceOperation>();
}  // namespace ttnn::prim
