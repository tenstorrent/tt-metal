// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "tanh_accurate.hpp"

#include "device/tanh_accurate_device_operation.hpp"

#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"

namespace ttnn {

namespace operations {

namespace unary {

namespace {
inline Tensor invoke_accurate_common(
    const Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    UnaryOpType op_type) {
    auto input_dtype = input_tensor.dtype();
    DataType output_dtype = input_dtype;

    TT_FATAL(
        input_dtype == DataType::BFLOAT16 || input_dtype == DataType::FLOAT32,
        "Supported dtypes for accurate operations is : BFLOAT16 or FLOAT32");

    bool preserve_fp32_precision = (input_dtype == DataType::FLOAT32);

    bool fp32_dest_acc_en = preserve_fp32_precision or output_dtype == DataType::UINT32 or
                            output_dtype == DataType::INT32 or output_dtype == DataType::FLOAT32 or
                            input_dtype == DataType::UINT32 or input_dtype == DataType::INT32;

    bool bfp8_pack_precise = input_dtype == DataType::BFLOAT8_B;

    auto output_memory_config = optional_output_tensor.has_value()
                                    ? optional_output_tensor.value().memory_config()
                                    : memory_config.value_or(input_tensor.memory_config());

    const std::vector<EltwiseUnaryWithParam>& op_chain = {EltwiseUnaryWithParam{op_type}};
    return prim::tanh_accurate(
        input_tensor,
        op_chain,
        output_dtype,
        output_memory_config,
        fp32_dest_acc_en,
        preserve_fp32_precision,
        bfp8_pack_precise,
        optional_output_tensor);
}
}  // namespace

Tensor Tanh_accurate::invoke(
    const Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return invoke_accurate_common(input_tensor, memory_config, optional_output_tensor, UnaryOpType::TANH);
}

Tensor Tanhshrink_accurate::invoke(
    const Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return invoke_accurate_common(input_tensor, memory_config, optional_output_tensor, UnaryOpType::TANHSHRINK);
}

}  // namespace unary
}  // namespace operations
}  // namespace ttnn
