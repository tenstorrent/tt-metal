// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_eager/tt_dnn/op_library/run_operation.hpp"
#include "tt_eager/tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"
namespace ttnn {

namespace operations {

namespace unary {

namespace detail {

inline Tensor run_eltwise_unary(const Tensor& input_tensor, std::vector<UnaryWithParam> ops_chain, const MemoryConfig& memory_config) {
    TT_FATAL(ops_chain.size() > 0, "At least 1 unary op must be specified");
    bool fp32_dest_acc_en = input_tensor.get_dtype() == DataType::UINT32 or input_tensor.get_dtype() == DataType::INT32; // MT: Currently only uint32/int32 is moved to DST directly, fp32 is converted to fp16b
    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input_tensor}))};
    operation::launch_op(
        [ops_chain, memory_config, fp32_dest_acc_en] (const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors) mutable -> std::vector<Tensor> {
            return operation::run(EltwiseUnary{ops_chain, memory_config, fp32_dest_acc_en}, input_tensors);
        }, {input_tensor}, output_tensors);
    return output_tensors.at(0);
}

template <UnaryOpType unary_op_type>
struct make_eltwise_unary {
    Tensor operator()( const Tensor& input_tensor, const MemoryConfig& memory_config) const {
        return detail::run_eltwise_unary(input_tensor, {UnaryWithParam{.op_type = unary_op_type}}, memory_config);
    }
};

constexpr auto silu = detail::make_eltwise_unary<UnaryOpType::SILU>{};

} // namespace detail


Tensor silu(const Tensor& input_tensor_a, const std::optional<MemoryConfig>& memory_config) {
    return detail::silu(input_tensor_a, memory_config.value_or(operation::DEFAULT_OUTPUT_MEMORY_CONFIG));
}

} // namespace unary
} // namespace operations
} // namespace ttnn

using ttnn::operations::unary::silu;
