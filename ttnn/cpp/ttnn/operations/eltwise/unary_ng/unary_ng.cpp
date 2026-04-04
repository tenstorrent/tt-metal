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

DEFINE_UNARY_NG_OP(cosh, COSH)
DEFINE_UNARY_NG_OP(cbrt, CBRT)

#undef DEFINE_UNARY_NG_OP

}  // namespace ttnn
