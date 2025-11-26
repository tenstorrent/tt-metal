// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "convert_to_hwc_program_factory.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/decorators.hpp"

#include "convert_to_hwc_device_operation_types.hpp"

namespace ttnn::operations::experimental::cnn {

struct ConvertToHWCDeviceOperation {
    using operation_attributes_t = cnn::operation_attributes_t;
    using tensor_args_t = cnn::tensor_args_t;
    using spec_return_value_t = cnn::spec_return_value_t;
    using tensor_return_value_t = cnn::tensor_return_value_t;
    using program_factory_t = std::variant<program::ConvertToHWCProgramFactory>;
    using shared_variables_t = program::ConvertToHWCProgramFactory::shared_variables_t;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);

    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input, const MemoryConfig& memory_config, const DataType& dtype);
};

}  // namespace ttnn::operations::experimental::cnn

namespace ttnn::prim {
constexpr auto convert_to_hwc = ttnn::register_operation<
    "ttnn::prim::convert_to_hwc",
    ttnn::operations::experimental::cnn::ConvertToHWCDeviceOperation>();
}  // namespace ttnn::prim
