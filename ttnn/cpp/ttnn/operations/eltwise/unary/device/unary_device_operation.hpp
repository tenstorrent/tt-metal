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

namespace ttnn::operations::unary {

struct UnaryDeviceOperation {
    using operation_attributes_t = unary::operation_attributes_t;
    using tensor_args_t = unary::tensor_args_t;
    using shape_return_value_t = unary::shape_return_value_t;
    using tensor_return_value_t = unary::tensor_return_value_t;
    using program_factory_t = std::variant<program::UnaryProgramFactory, program::UnaryShardedProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static shape_return_value_t compute_output_shapes(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t& operation_attributes,
                                                       const tensor_args_t&);

    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(const Tensor& input,
                                                                    const std::vector<UnaryWithParam>& op_chain,
                                                                    DataType output_dtype,
                                                                    const MemoryConfig& output_memory_config,
                                                                    bool fp32_dest_acc_en,
                                                                    bool preserve_fp32_precision,
                                                                    const std::optional<Tensor>& preallocated_output);
};

}  // namespace ttnn::operations::unary

namespace ttnn::prim {
constexpr auto unary = ttnn::register_operation<"ttnn::prim::unary", ttnn::operations::unary::UnaryDeviceOperation>();
}  // namespace ttnn::prim
