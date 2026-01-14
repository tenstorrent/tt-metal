// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "fill_pad_program_factory.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/decorators.hpp"

#include "fill_pad_device_operation_types.hpp"

namespace ttnn::operations::data_movement::fill_pad {

struct FillPadDeviceOperation {
    using operation_attributes_t = fill_pad::operation_attributes_t;
    using tensor_args_t = fill_pad::tensor_args_t;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<program::FillPadProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t& args, const tensor_args_t&);
};

}  // namespace ttnn::operations::data_movement::fill_pad

namespace ttnn::prim {
ttnn::Tensor fill_pad(const Tensor& input, float fill_value, const MemoryConfig& output_memory_config);
}  // namespace ttnn::prim
