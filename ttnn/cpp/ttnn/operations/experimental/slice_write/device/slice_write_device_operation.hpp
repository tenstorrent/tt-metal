// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/decorators.hpp"

#include "slice_write_device_operation_types.hpp"
#include "slice_write_rm_sharded_input_program_factory.hpp"
#include "slice_write_tiled_sharded_input_program_factory.hpp"
#include "slice_write_rm_interleaved_program_factory.hpp"

namespace ttnn::experimental::prim {

struct SliceWriteDeviceOperation {
    using operation_attributes_t = SliceWriteParams;
    using tensor_args_t = SliceWriteInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<
        SliceWriteRMShardedInputProgramFactory,
        SliceWriteTiledShardedInputProgramFactory,
        SliceWriteRMInterleavedProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);

    static tt::stl::hash::hash_t compute_program_hash(
        const operation_attributes_t& args, const tensor_args_t& tensor_args);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor slice_write(
    const Tensor& input_tensor,
    Tensor& output_tensor,
    const ttnn::Shape& slice_start,
    const ttnn::Shape& slice_end,
    const ttnn::Shape& step);

}  // namespace ttnn::prim
