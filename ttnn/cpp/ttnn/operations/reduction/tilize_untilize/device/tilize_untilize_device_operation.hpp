// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "tilize_untilize_program_factory.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/decorators.hpp"
#include "tilize_untilize_device_operation_types.hpp"

namespace ttnn::operations::reduction {

struct TilizeUntilizeDeviceOperation {
    using operation_attributes_t = ttnn::operations::reduction::operation_attributes_t;
    using tensor_args_t = ttnn::operations::reduction::tensor_args_t;
    using spec_return_value_t = ttnn::operations::reduction::spec_return_value_t;
    using tensor_return_value_t = ttnn::operations::reduction::tensor_return_value_t;
    using program_factory_t = std::variant<program::TilizeUntilizeProgramFactory>;

    // ALL STATIC FUNCTIONS - This is the modern pattern!
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);
    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input,
        std::optional<MemoryConfig> output_memory_config,
        std::optional<DataType> output_dtype,
        const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
        OpType op_type = OpType::IDENTITY,
        float scaler = 1.0f);
};

}  // namespace ttnn::operations::reduction

namespace ttnn::prim {
constexpr auto tilize_untilize = ttnn::
    register_operation<"ttnn::prim::tilize_untilize", ttnn::operations::reduction::TilizeUntilizeDeviceOperation>();
}  // namespace ttnn::prim
