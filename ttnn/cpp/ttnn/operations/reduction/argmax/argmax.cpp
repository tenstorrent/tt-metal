// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#include "device/argmax_device_operation.hpp"
#include "device/argmax_utils.hpp"
#include "ttnn/operations/reduction/argmax/argmax.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/decorators.hpp"

#include <utility>

namespace ttnn::operations::reduction {

/* Creates appropriate output tensor for a given zero volume input tensor.
   The output tensor has the same shape as the input tensor, except that the dimensions
   specified in dim are reduced to 1.
   The output tensor is filled with NAN values.
*/
static Tensor zero_volume_argmax(
    const Tensor& input_tensor, const std::optional<int>& dim, bool keepdim, const MemoryConfig& memory_config) {
    auto output_shape = ttnn::prim::get_output_shape(input_tensor, dim, keepdim);

    return ttnn::full(
        ttnn::Shape(output_shape),
        NAN,
        tt::tt_metal::DataType::UINT32,
        input_tensor.layout(),
        *input_tensor.device(),
        memory_config);
}

ttnn::Tensor ArgMaxOperation::invoke(
    const Tensor& input_tensor,
    const std::optional<int>& dim,
    bool keepdim,
    const std::optional<CoreRangeSet>& sub_core_grids,
    bool use_multicore,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor) {
    auto input_shape = input_tensor.logical_shape();
    auto output_memory_config = memory_config.value_or(input_tensor.memory_config());

    // If the input is a zero volume tensor, return output with shape adjusted for keepdim
    if (input_tensor.logical_volume() == 0) [[unlikely]] {
        return zero_volume_argmax(input_tensor, dim, keepdim, output_memory_config);
    }

    auto rank = input_shape.size();
    // If the input is a rank 0 tensor, return a rank 0 tensor
    if (rank == 0) [[unlikely]] {
        return ttnn::full(
            input_shape,
            /*fill_value=*/0,
            tt::tt_metal::DataType::UINT32,
            input_tensor.layout(),
            *input_tensor.device(),
            output_memory_config);
    }

    return ttnn::prim::argmax(
        input_tensor,
        tt::tt_metal::DataType::UINT32,
        dim,
        keepdim,
        sub_core_grids,
        use_multicore,
        output_memory_config,
        std::move(optional_output_tensor));
}

}  // namespace ttnn::operations::reduction
