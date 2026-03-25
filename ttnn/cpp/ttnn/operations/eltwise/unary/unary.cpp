// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "unary.hpp"

#include "common/unary_op_types.hpp"
#include "device/unary_device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/core/core.hpp"
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

Tensor mac_tss(
    const Tensor& input_tensor_a,
    float value1,
    float value2,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    return ttnn::detail::unary_impl(
        input_tensor_a,
        {EltwiseUnaryWithParam{UnaryOpType::MAC_TSS, std::vector<float>{value1, value2}}},
        memory_config,
        optional_output_tensor,
        sub_core_grids);
}

}  // namespace ttnn
