// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "tanh_accurate.hpp"

#include "device/tanh_accurate_device_operation.hpp"
#include "ttnn/common/queue_id.hpp"

namespace ttnn {

namespace operations {

namespace unary {

Tensor Tanh_accurate::invoke(
    QueueId queue_id,
    const Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    TT_FATAL(
        input_tensor.dtype() == DataType::BFLOAT16,
        "Supported dtypes for tanh with accuracy mode enabled is : BFLOAT16");

    auto input_dtype = input_tensor.dtype();
    DataType output_dtype = input_dtype;

    bool preserve_fp32_precision = (input_dtype == DataType::FLOAT32);

    bool fp32_dest_acc_en = preserve_fp32_precision or output_dtype == DataType::UINT32 or
                            output_dtype == DataType::INT32 or output_dtype == DataType::FLOAT32 or
                            input_dtype == DataType::UINT32 or input_dtype == DataType::INT32;

    bool bfp8_pack_precise = input_dtype == DataType::BFLOAT8_B;

    auto output_memory_config = optional_output_tensor.has_value()
                                    ? optional_output_tensor.value().memory_config()
                                    : memory_config.value_or(input_tensor.memory_config());

    return prim::tanh_accurate(
        queue_id,
        input_tensor,
        output_dtype,
        output_memory_config,
        fp32_dest_acc_en,
        preserve_fp32_precision,
        bfp8_pack_precise,
        optional_output_tensor);
}

}  // namespace unary
}  // namespace operations
}  // namespace ttnn
