// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "i0_fpu.hpp"

#include "device/i0_fpu_device_operation.hpp"
#include "ttnn/common/queue_id.hpp"

namespace ttnn {

namespace operations {

namespace unary {

Tensor I0Fpu::invoke(
    QueueId queue_id,
    const Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    TT_FATAL(input_tensor.dtype() == DataType::BFLOAT16, "Supported dtypes for i0_fpu is : BFLOAT16");

    auto input_dtype = input_tensor.dtype();
    DataType output_dtype = input_dtype;

    bool preserve_fp32_precision = (input_dtype == DataType::FLOAT32);

    bool fp32_dest_acc_en = preserve_fp32_precision || output_dtype == DataType::UINT32 ||
                            output_dtype == DataType::INT32 || output_dtype == DataType::FLOAT32 ||
                            input_dtype == DataType::UINT32 || input_dtype == DataType::INT32;

    bool bfp8_pack_precise = input_dtype == DataType::BFLOAT8_B;

    auto output_memory_config = optional_output_tensor.has_value()
                                    ? optional_output_tensor.value().memory_config()
                                    : memory_config.value_or(input_tensor.memory_config());

    return prim::i0_fpu(
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
