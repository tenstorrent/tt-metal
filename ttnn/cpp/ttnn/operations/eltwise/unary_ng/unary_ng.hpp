// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/eltwise/complex/complex.hpp"

namespace ttnn::operations::unary_ng::detail {

Tensor unary_ng_impl(
    const Tensor& input_tensor,
    const std::vector<unary::EltwiseUnaryWithParam>& op_chain,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

}  // namespace ttnn::operations::unary_ng::detail

namespace ttnn {

#define DECLARE_UNARY_NG_OP(op_name)                                                   \
    Tensor op_name(                                                                    \
        const Tensor& input_tensor,                                                    \
        const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt, \
        const std::optional<Tensor>& optional_output_tensor = std::nullopt,            \
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

DECLARE_UNARY_NG_OP(abs)
DECLARE_UNARY_NG_OP(neg)
DECLARE_UNARY_NG_OP(acos)
DECLARE_UNARY_NG_OP(asin)
DECLARE_UNARY_NG_OP(asinh)
DECLARE_UNARY_NG_OP(atan)
DECLARE_UNARY_NG_OP(atanh)
DECLARE_UNARY_NG_OP(cos)
DECLARE_UNARY_NG_OP(acosh)
DECLARE_UNARY_NG_OP(cosh)
DECLARE_UNARY_NG_OP(sinh)
DECLARE_UNARY_NG_OP(erfinv)
DECLARE_UNARY_NG_OP(exp2)
DECLARE_UNARY_NG_OP(expm1)
DECLARE_UNARY_NG_OP(gez)
DECLARE_UNARY_NG_OP(gtz)
DECLARE_UNARY_NG_OP(i0)
DECLARE_UNARY_NG_OP(i1)
DECLARE_UNARY_NG_OP(isfinite)
DECLARE_UNARY_NG_OP(isinf)
DECLARE_UNARY_NG_OP(isnan)
DECLARE_UNARY_NG_OP(isneginf)
DECLARE_UNARY_NG_OP(isposinf)
DECLARE_UNARY_NG_OP(lez)
DECLARE_UNARY_NG_OP(logical_not)
DECLARE_UNARY_NG_OP(ltz)
DECLARE_UNARY_NG_OP(nez)
DECLARE_UNARY_NG_OP(reciprocal)
DECLARE_UNARY_NG_OP(relu)
DECLARE_UNARY_NG_OP(relu6)
DECLARE_UNARY_NG_OP(sign)
DECLARE_UNARY_NG_OP(signbit)
DECLARE_UNARY_NG_OP(silu)
DECLARE_UNARY_NG_OP(sin)
DECLARE_UNARY_NG_OP(square)
DECLARE_UNARY_NG_OP(tan)
DECLARE_UNARY_NG_OP(tiled_prod)
DECLARE_UNARY_NG_OP(bitwise_not)
DECLARE_UNARY_NG_OP(alt_complex_rotate90)
DECLARE_UNARY_NG_OP(floor)
DECLARE_UNARY_NG_OP(ceil)
DECLARE_UNARY_NG_OP(trunc)
DECLARE_UNARY_NG_OP(frac)
DECLARE_UNARY_NG_OP(hardsigmoid)
DECLARE_UNARY_NG_OP(hardswish)
DECLARE_UNARY_NG_OP(softsign)
DECLARE_UNARY_NG_OP(cbrt)
DECLARE_UNARY_NG_OP(lgamma)
DECLARE_UNARY_NG_OP(digamma)

#undef DECLARE_UNARY_NG_OP

Tensor abs(const ComplexTensor& input_tensor, const tt::tt_metal::MemoryConfig& output_mem_config);

}  // namespace ttnn
