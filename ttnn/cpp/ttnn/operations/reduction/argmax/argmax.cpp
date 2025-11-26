// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/reduction/argmax/argmax.hpp"

#include <utility>

#include "device/argmax_device_operation.hpp"

#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation.hpp"

namespace ttnn::operations::reduction {

namespace {

// Helper to generate output shape for the reduction operation (for edge cases)
ttnn::SmallVector<uint32_t> get_output_shape(const Tensor& input_tensor, const std::optional<int>& dim, bool keepdim) {
    auto input_shape = input_tensor.logical_shape();
    int rank = input_shape.size();
    ttnn::SmallVector<uint32_t> output_shape;

    auto all_dim_reduce = !dim.has_value();
    auto red_dim = dim.value_or(0);

    if (rank > 0 && !((red_dim >= -rank) && (red_dim < rank))) {
        TT_THROW("Invalid reduction dimension {} for input tensor with rank {}", red_dim, rank);
    }

    red_dim = red_dim < 0 ? red_dim + rank : red_dim;

    for (int d = 0; d < rank; ++d) {
        bool is_reduction_dim = all_dim_reduce || (d == red_dim);

        if (is_reduction_dim) {
            if (keepdim) {
                output_shape.push_back(1);
            }
        } else {
            output_shape.push_back(input_shape[d]);
        }
    }

    return output_shape;
}

// Creates appropriate output tensor for a given zero volume input tensor.
// The output tensor has the same shape as the input tensor, except that the dimensions
// specified in dim are reduced to 1.
// The output tensor is filled with NAN values.
Tensor zero_volume_argmax(
    const Tensor& input_tensor, const std::optional<int> dim, const bool keepdim, const MemoryConfig& memory_config) {
    auto output_shape = get_output_shape(input_tensor, dim, keepdim);

    return ttnn::full(
        ttnn::Shape(output_shape),
        NAN,
        tt::tt_metal::DataType::UINT32,
        input_tensor.layout(),
        *input_tensor.device(),
        memory_config);
}

}  // namespace

ttnn::Tensor ArgMaxOperation::invoke(
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
            *input_tensor.device(),
            output_memory_config);
    }

    return ttnn::prim::argmax(
        input_tensor, dim, keepdim, sub_core_grids, use_muticore, memory_config, std::move(optional_output_tensor));
}

}  // namespace ttnn::operations::reduction
