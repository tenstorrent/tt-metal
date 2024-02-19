// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_eager/tt_numpy/functions.hpp"
#include "ttnn/types.hpp"

namespace ttnn {
namespace operations {
namespace core {

inline ttnn::Tensor reshape(const ttnn::Tensor& tensor, const ttnn::Shape& shape) {
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
    return ttnn::operations::core::reshape(tensor, tensor_shape_4D);
}

inline ttnn::Tensor from_device(const ttnn::Tensor& tensor) { return tensor.cpu(); }

// TODO : @eyonland move these creation functions to creation.hpp
template <typename T>
inline ttnn::Tensor full(
    const ttnn::Shape& shape,
    const T value,
    const DataType data_type,
    const Layout layout,
    Device& device,
    const MemoryConfig& output_mem_config = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) {
    return tt::numpy::full(shape.with_tile_padding().value(), value, data_type, layout, &device, output_mem_config);
}

inline ttnn::Tensor zeros(
    const ttnn::Shape& shape,
    const DataType data_type,
    const Layout layout,
    Device& device,
    const MemoryConfig& output_mem_config = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) {
    return full(shape, 0.0f, data_type, layout, device, output_mem_config);
}

inline ttnn::Tensor ones(
    const ttnn::Shape& shape,
    const DataType data_type,
    const Layout layout,
    Device& device,
    const MemoryConfig& output_mem_config = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) {
    return full(shape, 1.0f, data_type, layout, device, output_mem_config);
}

}  // namespace core
}  // namespace operations

using operations::core::from_device;
using operations::core::full;
using operations::core::ones;
using operations::core::reshape;
using operations::core::unsqueeze_to_4D;
using operations::core::zeros;

}  // namespace ttnn
