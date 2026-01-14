// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/decorators.hpp"

#include "ttnn/operations/reduction/moe/device/moe_device_operation_types.hpp"
#include "ttnn/operations/reduction/moe/device/moe_program_factory.hpp"

namespace ttnn::operations::reduction::moe {

struct MoeDeviceOperation {
    using operation_attributes_t = moe::operation_attributes_t;
    using tensor_args_t = moe::tensor_args_t;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<program::MoeProgramFactory>;
    using shared_variables_t = program::MoeProgramFactory::shared_variables_t;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::reduction::moe

namespace ttnn::prim {
ttnn::Tensor moe(
    const Tensor& input_tensor,
    const Tensor& expert_mask_tensor,
    const Tensor& topk_mask_tensor,
    uint16_t k,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& preallocated_output_tensor = std::nullopt);
}  // namespace ttnn::prim
