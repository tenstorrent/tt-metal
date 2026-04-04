// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "unary.hpp"

#include "common/unary_op_types.hpp"
#include "device/unary_device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/eltwise/complex/complex.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/binary/binary_composite.hpp"
#include "ttnn/operations/eltwise/ternary/ternary.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/tensor/tensor.hpp"

using namespace ttnn::operations::unary;

namespace ttnn::detail {

Tensor unary_impl(
    const Tensor& input_tensor,
    const std::vector<EltwiseUnaryWithParam>& op_chain,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    TT_FATAL(!op_chain.empty(), "Op chain cannot be empty");
    DataType input_dtype = input_tensor.dtype();
    // TYPECAST/BITCAST should always be the last operation in the chain when present; use its output dtype (param 1)
    DataType output_dtype = input_dtype;
    if (op_chain.back().type() == UnaryOpType::TYPECAST || op_chain.back().type() == UnaryOpType::BITCAST) {
        output_dtype = static_cast<DataType>(*op_chain.back().get_param_if<float>(1));
    }
    bool preserve_fp32_precision = input_dtype == DataType::FLOAT32;
    bool fp32_dest_acc_en = preserve_fp32_precision or output_dtype == DataType::UINT32 or
                            output_dtype == DataType::INT32 or output_dtype == DataType::FLOAT32 or
                            output_dtype == DataType::UINT8 or input_dtype == DataType::UINT8 or
                            input_dtype == DataType::UINT32 or input_dtype == DataType::INT32;
    bool bfp8_pack_precise = (op_chain.back().type() == UnaryOpType::TYPECAST && output_dtype == DataType::BFLOAT8_B);

    auto output_memory_config = optional_output_tensor.has_value()
                                    ? optional_output_tensor.value().memory_config()
                                    : memory_config.value_or(input_tensor.memory_config());
    return prim::unary(
        input_tensor,
        op_chain,
        output_dtype,
        output_memory_config,
        fp32_dest_acc_en,
        preserve_fp32_precision,
        bfp8_pack_precise,
        optional_output_tensor,
        sub_core_grids);
}

}  // namespace ttnn::detail

namespace ttnn {

Tensor identity(
    const Tensor& input_tensor,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return ttnn::detail::unary_impl(
        input_tensor, {UnaryWithParam{UnaryOpType::IDENTITY}}, memory_config, optional_output_tensor);
}

Tensor logit(
    const Tensor& input_tensor,
    std::optional<float> eps,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return ttnn::detail::unary_impl(
        input_tensor,
        {UnaryWithParam{UnaryOpType::LOGIT, {eps.value_or(-1.0f)}}},
        memory_config,
        optional_output_tensor);
}

Tensor unary_chain(
    const Tensor& input_tensor,
    const std::vector<EltwiseUnaryWithParam>& ops_chain,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return ttnn::detail::unary_impl(input_tensor, ops_chain, memory_config, optional_output_tensor);
}

Tensor unary_with_int32_param(
    UnaryOpType op_type,
    const Tensor& input_tensor,
    int32_t param,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return ttnn::detail::unary_impl(
        input_tensor, {EltwiseUnaryWithParam{op_type, param}}, memory_config, optional_output_tensor);
}

Tensor sub_sfpu(
    const Tensor& input_a,
    const Tensor& input_b,
    const std::optional<tt::tt_metal::MemoryConfig>& /*memory_config*/,
    const std::optional<Tensor>& /*optional_output_tensor*/) {
    // Stub: binary sub_sfpu (Tensor - Tensor) not yet re-implemented
    // Use ttnn::subtract instead
    return ttnn::subtract(input_a, input_b);
}

Tensor sub_sfpu(
    float lhs,
    const Tensor& rhs,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& /*optional_output_tensor*/) {
    // lhs - rhs = lhs + (-rhs) = -rhs + lhs
    return ttnn::add(ttnn::neg(rhs, memory_config), lhs, std::nullopt, memory_config);
}

Tensor where_tss(
    const Tensor& condition,
    float t_true,
    float t_false,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    return ttnn::detail::unary_impl(
        condition,
        {UnaryWithParam{UnaryOpType::WHERE_TSS, {t_true, t_false}}},
        memory_config,
        optional_output_tensor,
        sub_core_grids);
}

operations::complex::ComplexTensor reciprocal(
    const operations::complex::ComplexTensor& /*input*/, const tt::tt_metal::MemoryConfig& /*mem_config*/) {
    // Stub: complex reciprocal not yet re-implemented after nuke
    TT_THROW("Complex reciprocal not yet re-implemented");
}

Tensor abs(
    const operations::complex::ComplexTensor& /*input_tensor*/,
    const tt::tt_metal::MemoryConfig& /*output_mem_config*/) {
    // Stub: complex abs not yet re-implemented after nuke
    TT_THROW("Complex abs not yet re-implemented");
}

Tensor sigmoid(
    const Tensor& input_tensor,
    [[maybe_unused]] int vector_mode,
    operations::unary::SigmoidMode mode,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    bool fast_and_approximate_mode = (mode == operations::unary::SigmoidMode::FAST_APPROXIMATE);
    return ttnn::detail::unary_impl(
        input_tensor,
        {UnaryWithParam{UnaryOpType::SIGMOID, static_cast<float>(fast_and_approximate_mode)}},
        memory_config,
        optional_output_tensor);
}

Tensor softplus(
    const Tensor& input_tensor,
    float beta,
    float threshold,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return ttnn::detail::unary_impl(
        input_tensor,
        {UnaryWithParam{UnaryOpType::SOFTPLUS, {beta, threshold}}},
        memory_config,
        optional_output_tensor);
}

Tensor xielu(
    const Tensor& input_tensor,
    float alpha_p,
    float alpha_n,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return ttnn::detail::unary_impl(
        input_tensor, {UnaryWithParam{UnaryOpType::XIELU, {alpha_p, alpha_n}}}, memory_config, optional_output_tensor);
}

Tensor clamp_tss(
    const Tensor& input_tensor,
    float min_val,
    float max_val,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return ttnn::detail::unary_impl(
        input_tensor,
        {UnaryWithParam{UnaryOpType::CLAMP_TSS, {min_val, max_val}}},
        memory_config,
        optional_output_tensor);
}

}  // namespace ttnn
