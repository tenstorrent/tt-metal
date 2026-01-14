// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>
#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "padded_slice_device_operation_types.hpp"
#include "padded_slice_rm_program_factory.hpp"
#include "padded_slice_tile_program_factory.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/decorators.hpp"

namespace ttnn::operations::experimental::padded_slice {

struct PaddedSliceDeviceOperation {
    using operation_attributes_t = padded_slice::operation_attributes_t;
    using tensor_args_t = padded_slice::tensor_args_t;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t =
        std::variant<program::PaddedSliceRMProgramFactory, program::PaddedSliceTileProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);

    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::padded_slice

namespace ttnn::prim {

ttnn::operations::experimental::padded_slice::PaddedSliceDeviceOperation::tensor_return_value_t padded_slice(
    const Tensor& input,
    const ttnn::Shape& padded_slice_start,
    const ttnn::Shape& padded_slice_end,
    const ttnn::Shape& step,
    const MemoryConfig& output_mem_config,
    const std::optional<Tensor>& preallocated_output = std::nullopt);

}  // namespace ttnn::prim
