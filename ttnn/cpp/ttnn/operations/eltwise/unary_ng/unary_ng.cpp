// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "unary_ng.hpp"
#include "device/unary_ng_device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/eltwise/complex/complex.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"

namespace ttnn::operations::unary_ng::detail {

Tensor unary_ng_impl(
    const Tensor& input_tensor,
    const std::vector<unary::EltwiseUnaryWithParam>& op_chain,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    TT_FATAL(!op_chain.empty(), "Op chain cannot be empty");
    DataType input_dtype = input_tensor.dtype();
    DataType output_dtype = input_dtype;
    if (op_chain.back().type() == unary::UnaryOpType::TYPECAST ||
        op_chain.back().type() == unary::UnaryOpType::BITCAST) {
        output_dtype = static_cast<DataType>(*op_chain.back().get_param_if<float>(1));
    }
    bool preserve_fp32_precision = (input_dtype == DataType::FLOAT32);
    bool fp32_dest_acc_en = preserve_fp32_precision || output_dtype == DataType::UINT32 ||
                            output_dtype == DataType::INT32 || output_dtype == DataType::FLOAT32 ||
                            output_dtype == DataType::UINT8 || input_dtype == DataType::UINT8 ||
                            input_dtype == DataType::UINT32 || input_dtype == DataType::INT32;
    bool bfp8_pack_precise =
        (op_chain.back().type() == unary::UnaryOpType::TYPECAST && output_dtype == DataType::BFLOAT8_B);

    auto output_memory_config = optional_output_tensor.has_value()
                                    ? optional_output_tensor.value().memory_config()
                                    : memory_config.value_or(input_tensor.memory_config());

    return prim::unary_ng(
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

}  // namespace ttnn::operations::unary_ng::detail

namespace ttnn {

Tensor abs(
    const Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    using namespace operations::unary;
    UnaryOpType op_type = UnaryOpType::ABS;
    if (input_tensor.dtype() == DataType::INT32) {
        op_type = UnaryOpType::ABS_INT32;
    }
    return operations::unary_ng::detail::unary_ng_impl(
        input_tensor, {UnaryWithParam{op_type}}, memory_config, optional_output_tensor, sub_core_grids);
}

Tensor abs(const ComplexTensor& input_tensor, const MemoryConfig& output_mem_config) {
    return ttnn::hypot(input_tensor[0], input_tensor[1], output_mem_config);
}

// Helper macro: most basic ops just forward a single UnaryOpType to unary_ng_impl.
#define DEFINE_UNARY_NG_OP(op_name, OP_TYPE)                 \
    Tensor op_name(                                          \
        const Tensor& input_tensor,                          \
        const std::optional<MemoryConfig>& memory_config,    \
        const std::optional<Tensor>& optional_output_tensor, \
        const std::optional<CoreRangeSet>& sub_core_grids) { \
        using namespace operations::unary;                   \
        return operations::unary_ng::detail::unary_ng_impl(  \
            input_tensor,                                    \
            {UnaryWithParam{UnaryOpType::OP_TYPE}},          \
            memory_config,                                   \
            optional_output_tensor,                          \
            sub_core_grids);                                 \
    }

DEFINE_UNARY_NG_OP(neg, NEG)
DEFINE_UNARY_NG_OP(acos, ACOS)
DEFINE_UNARY_NG_OP(asin, ASIN)
DEFINE_UNARY_NG_OP(asinh, ASINH)
DEFINE_UNARY_NG_OP(atan, ATAN)
DEFINE_UNARY_NG_OP(atanh, ATANH)
DEFINE_UNARY_NG_OP(cos, COS)
DEFINE_UNARY_NG_OP(acosh, ACOSH)
DEFINE_UNARY_NG_OP(cosh, COSH)
DEFINE_UNARY_NG_OP(sinh, SINH)
DEFINE_UNARY_NG_OP(erfinv, ERFINV)
DEFINE_UNARY_NG_OP(exp2, EXP2)
DEFINE_UNARY_NG_OP(expm1, EXPM1)
DEFINE_UNARY_NG_OP(gez, GEZ)
DEFINE_UNARY_NG_OP(gtz, GTZ)
DEFINE_UNARY_NG_OP(i0, I0)
DEFINE_UNARY_NG_OP(i1, I1)
DEFINE_UNARY_NG_OP(isfinite, ISFINITE)
DEFINE_UNARY_NG_OP(isinf, ISINF)
DEFINE_UNARY_NG_OP(isnan, ISNAN)
DEFINE_UNARY_NG_OP(isneginf, ISNEGINF)
DEFINE_UNARY_NG_OP(isposinf, ISPOSINF)
DEFINE_UNARY_NG_OP(lez, LEZ)
DEFINE_UNARY_NG_OP(logical_not, LOGICAL_NOT_UNARY)
DEFINE_UNARY_NG_OP(ltz, LTZ)
DEFINE_UNARY_NG_OP(nez, NEZ)
DEFINE_UNARY_NG_OP(reciprocal, RECIP)
DEFINE_UNARY_NG_OP(relu, RELU)
DEFINE_UNARY_NG_OP(relu6, RELU6)
DEFINE_UNARY_NG_OP(sign, SIGN)
DEFINE_UNARY_NG_OP(signbit, SIGNBIT)
DEFINE_UNARY_NG_OP(silu, SILU)
DEFINE_UNARY_NG_OP(sin, SIN)
DEFINE_UNARY_NG_OP(square, SQUARE)
DEFINE_UNARY_NG_OP(tan, TAN)
DEFINE_UNARY_NG_OP(tiled_prod, TILED_PROD)
DEFINE_UNARY_NG_OP(bitwise_not, BITWISE_NOT)
DEFINE_UNARY_NG_OP(alt_complex_rotate90, ALT_COMPLEX_ROTATE90)
DEFINE_UNARY_NG_OP(floor, FLOOR)
DEFINE_UNARY_NG_OP(ceil, CEIL)
DEFINE_UNARY_NG_OP(trunc, TRUNC)
DEFINE_UNARY_NG_OP(frac, FRAC)
DEFINE_UNARY_NG_OP(hardsigmoid, HARDSIGMOID)
DEFINE_UNARY_NG_OP(hardswish, HARDSWISH)
DEFINE_UNARY_NG_OP(softsign, SOFTSIGN)
DEFINE_UNARY_NG_OP(cbrt, CBRT)
DEFINE_UNARY_NG_OP(lgamma, LGAMMA)

#undef DEFINE_UNARY_NG_OP

}  // namespace ttnn
