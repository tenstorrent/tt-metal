// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device/argmax_op.hpp"
#include "ttnn/operations/reduction/argmax/argmax.hpp"

#include <utility>

#include "ttnn/run_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation.hpp"

namespace ttnn::operations::reduction {

/* Creates appropriate output tensor for a given zero volume input tensor.
   The output tensor has the same shape as the input tensor, except that the dimensions
   specified in dim are reduced to 1.
   The output tensor is filled with NAN values.
*/
static Tensor zero_volume_argmax(
    const Tensor& input_tensor, const std::optional<int> dim, const bool keepdim, const MemoryConfig& memory_config) {
    auto input_shape = input_tensor.logical_shape();

    auto argmax_op = ArgMax{
        tt::tt_metal::DataType::UINT32,
        dim,
        keepdim,
        /*sub_core_grids=*/std::nullopt,
        /*use_multicore=*/false,
        /*output_mem_config=*/tt::tt_metal::MemoryConfig()};
    auto output_shape = argmax_op.get_output_shape(input_tensor);

    return ttnn::full(
        ttnn::Shape(output_shape),
        NAN,
        tt::tt_metal::DataType::UINT32,
        input_tensor.layout(),
        *input_tensor.mesh_device(),
        memory_config);
}

ttnn::Tensor ArgMaxOperation::invoke(
    QueueId queue_id,
    const Tensor& input_tensor,
    const std::optional<int> dim,
    const bool keepdim,
    const std::optional<CoreRangeSet>& sub_core_grids,
    const bool use_muticore,
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
            *input_tensor.mesh_device(),
            output_memory_config);
    }

    return tt::tt_metal::operation::run(
               ArgMax{tt::tt_metal::DataType::UINT32, dim, keepdim, sub_core_grids, use_muticore, output_memory_config},
               {input_tensor},
               {},
               {std::move(optional_output_tensor)},
               queue_id)
        .at(0);
}

}  // namespace ttnn::operations::reduction
