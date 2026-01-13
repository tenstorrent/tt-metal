// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "row_mean_sub_square_reduce_program_factory.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/decorators.hpp"
#include "row_mean_sub_square_reduce_device_operation_types.hpp"

namespace ttnn::operations::reduction::row_mean_sub_square_reduce {

struct RowMeanSubSquareReduceDeviceOperation {
    using operation_attributes_t = row_mean_sub_square_reduce::operation_attributes_t;
    using tensor_args_t = row_mean_sub_square_reduce::tensor_args_t;
    using spec_return_value_t = row_mean_sub_square_reduce::spec_return_value_t;
    using tensor_return_value_t = row_mean_sub_square_reduce::tensor_return_value_t;
    using program_factory_t = std::variant<program::RowMeanSubSquareReduceProgramFactory>;

    // ALL STATIC FUNCTIONS - This is the modern pattern!
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);
    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::reduction::row_mean_sub_square_reduce

namespace ttnn::prim {
// Primitive operation - free function that calls launch_on_device
ttnn::operations::reduction::row_mean_sub_square_reduce::RowMeanSubSquareReduceDeviceOperation::tensor_return_value_t
row_mean_sub_square_reduce(
    const Tensor& input, std::optional<DataType> output_dtype, const tt::tt_metal::MemoryConfig& output_mem_config);
}  // namespace ttnn::prim
