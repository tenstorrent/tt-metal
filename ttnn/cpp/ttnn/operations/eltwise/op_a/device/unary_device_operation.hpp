// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "unary_program_factory.hpp"
#include "unary_sharded_program_factory.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/decorators.hpp"

#include "unary_device_operation_types.hpp"


namespace ttnn::operations::op_a {

using ttnn::operations::unary::UnaryWithParam;
using ttnn::operations::unary::UnaryOpType;

struct UnaryDeviceOperation {

    using operation_attributes_t = op_a::operation_attributes_t;
    using tensor_args_t = op_a::tensor_args_t;
    using shape_return_value_t = op_a::shape_return_value_t;
    using tensor_return_value_t = op_a::tensor_return_value_t;
    using program_factory_t = std::variant<program::UnaryProgramFactory, program::UnaryShardedProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static shape_return_value_t compute_output_shapes(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t& operation_attributes, const tensor_args_t&);

    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input,
        const std::vector<UnaryWithParam>& op_chain,
        float param,
        DataType output_dtype,
        const MemoryConfig& output_memory_config,
        bool fp32_dest_acc_en,
        bool preserve_fp32_precision,
        const std::optional<Tensor>& preallocated_output);
};

}  // namespace ttnn::operations::op_a

// Register the operation with the ttnn::register_operation API to make it available to the user as
// ttnn::prim::op_a
namespace ttnn::prim {
constexpr auto op_a = ttnn::register_operation<"ttnn::prim::op_a", ttnn::operations::op_a::UnaryDeviceOperation>();
} // namespace ttnn::prim
