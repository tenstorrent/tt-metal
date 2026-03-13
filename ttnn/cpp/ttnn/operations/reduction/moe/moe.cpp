// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moe.hpp"

#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/data_movement/copy/copy.hpp"
#include "ttnn/operations/reduction/moe/device/moe_device_operation.hpp"

namespace ttnn {

Tensor moe(
    const Tensor& input_tensor,
    const Tensor& expert_mask_tensor,
    const Tensor& topk_mask_tensor,
    uint16_t k,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    std::optional<Tensor> output_tensor) {
    const auto& input_shape = input_tensor.logical_shape();

    // Zero-volume input: return zero-volume tensor with correct output shape
    // Note: no explicit handling of scalar input is done, because MOE requires 4D tensors
    // (a limitation specified in the ttnn.moe documentation).
    if (input_tensor.logical_volume() == 0) {
        ttnn::Shape desired_output_shape = input_shape;
        // The last dimension needs to be reduced to 1 to match the output shape of the torch reference.
        desired_output_shape[-1] = 1;
        if (!output_tensor.has_value()) {
            return ttnn::full(
                desired_output_shape,
                0.0f,
                input_tensor.dtype(),
                input_tensor.layout(),
                *input_tensor.device(),
                memory_config.value_or(input_tensor.memory_config()));
        }

        Tensor& preallocated_tensor = output_tensor.value();
        TT_FATAL(is_device_tensor(preallocated_tensor), "Preallocated output tensor must be on device");

        TT_FATAL(
            preallocated_tensor.logical_shape() == desired_output_shape,
            "Preallocated output tensor has incorrect shape! Got : {}, expected: {}",
            preallocated_tensor.logical_shape(),
            desired_output_shape);

        return output_tensor.value();
    }

    return ttnn::prim::moe(input_tensor, expert_mask_tensor, topk_mask_tensor, k, memory_config, output_tensor);
}

}  // namespace ttnn
