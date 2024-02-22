// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_dnn/op_library/eltwise_binary/eltwise_binary_op.hpp"
#include "tt_eager/tensor/tensor_utils.hpp"
#include "tt_eager/tt_dnn/op_library/bcast/bcast_op.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"

namespace ttnn {
namespace operations {
namespace core {

static const auto DRAM_MEMORY_CONFIG = tt::tt_metal::MemoryConfig{
    .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED, .buffer_type = tt::tt_metal::BufferType::DRAM};
static const auto L1_MEMORY_CONFIG = tt::tt_metal::MemoryConfig{
    .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED, .buffer_type = tt::tt_metal::BufferType::L1};

inline ttnn::Tensor reshape(const ttnn::Tensor& tensor, const ttnn::Shape& shape) {
    // TODO: make it work just like in python
    return ttnn::Tensor(tensor.reshape(shape.value()));
}

inline ttnn::Tensor unsqueeze_to_4D(const ttnn::Tensor& tensor) {
    const auto tensor_shape = tensor.ttnn_shape();
    const auto rank = tensor_shape.rank();
    if (rank == 4) {
        return tensor;
    }
    if (rank > 4) {
        TT_THROW("Tensor rank is greater than 4");
    }

    const auto tensor_shape_4D = tensor_shape.to_rank<4>();
    return reshape(tensor, tensor_shape_4D);
}

}  // namespace core
}  // namespace operations

using operations::core::reshape;
using operations::core::unsqueeze_to_4D;

}  // namespace ttnn
