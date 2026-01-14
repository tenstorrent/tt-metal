// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operation.hpp"
#include "expand.hpp"
#include <functional>
#include <ttnn/operations/functions.hpp>
#include "ttnn/operations/data_movement/repeat/repeat.hpp"
namespace ttnn::operations::expand {


ttnn::SmallVector<uint32_t> create_repetition_vector(const Tensor& tensor, std::span<const int32_t> shape) {
    ttnn::SmallVector<uint32_t> expansion_vector(shape.size());
    auto tensor_shape = tensor.logical_shape();
    const auto source_rank = tensor_shape.rank();
    const auto new_rank = shape.size();
    TT_FATAL(source_rank <= new_rank, "Only size 1 dimensions can be expanded in the output shape");
    for (auto index = 0; index < new_rank; ++index) {
        if (index >= source_rank) {
            expansion_vector[index] = shape[index];
        } else if ((shape[index] == -1) || (shape[index] == tensor_shape[index])) {
            expansion_vector[index] = 1;
        } else {
            TT_FATAL(tensor_shape[index] == 1, "Only size 1 dimensions can be expanded in the output shape");
            expansion_vector[index] = shape[index];
        }
    }
    return expansion_vector;
}

ttnn::Tensor ExpandOperation::invoke(
    const ttnn::Tensor& tensor,
    const tt::stl::Span<const int32_t> shape_vector,
    const std::optional<MemoryConfig>& memory_config) {
    return ttnn::repeat(tensor, create_repetition_vector(tensor, shape_vector), memory_config);
}

}  // namespace ttnn::operations::expand
