// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "prod_op_all.hpp"
#include "prod_all_device_operation.hpp"

#include <ttnn/operations/functions.hpp>
#include "ttnn/tensor/tensor_ops.hpp"

namespace tt::operations::primary {

tt::tt_metal::Tensor prod_all(const tt::tt_metal::Tensor& input, const tt::tt_metal::MemoryConfig& output_mem_config) {
    tt::tt_metal::Tensor result = ttnn::prim::prod_all(input, output_mem_config);
    switch (result.dtype()) {
        case tt::tt_metal::DataType::FLOAT32:
            return ttnn::prod_result_computation_WH_B0<float>(
                result, result.dtype(), result.layout(), result.device(), output_mem_config);
        case tt::tt_metal::DataType::BFLOAT16:
            return ttnn::prod_result_computation_WH_B0<bfloat16>(
                result, result.dtype(), result.layout(), result.device(), output_mem_config);
        case tt::tt_metal::DataType::BFLOAT8_B:
        case tt::tt_metal::DataType::BFLOAT4_B: {
            // Block-float tiles share one exponent per 16 elements, so a single value cannot be read
            // as a C++ scalar via host_buffer::get_as. Unpack to FLOAT32 on host first (the same path
            // ttnn.to_torch() uses) and reuse the float computation.
            auto* device = result.device();
            const tt::tt_metal::Tensor result_fp32 =
                tt::tt_metal::to_dtype(result.cpu(), tt::tt_metal::DataType::FLOAT32);
            return ttnn::prod_result_computation_WH_B0<float>(
                result_fp32, tt::tt_metal::DataType::FLOAT32, result_fp32.layout(), device, output_mem_config);
        }
        default:
            TT_THROW(
                "Error - unsupported data type for prod, expected BFLOAT16, FLOAT32, BFLOAT8_B or BFLOAT4_B but got "
                "{}.",
                result.dtype());
    }
}

}  // namespace tt::operations::primary
