// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_eager/tensor/tensor.hpp"
#include "tt_eager/tensor/tensor_utils.hpp"
#include "tt_eager/tensor/types.hpp"
#include "tt_eager/tt_numpy/functions.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "ttnn/types.hpp"
#include "ttnn/validation.hpp"

namespace ttnn {
namespace operations {
namespace creation {

template <typename T>
inline ttnn::Tensor full(
    const ttnn::Shape& shape,
    const T value,
    const DataType data_type,
    const Layout layout,
    Device& device,
    const MemoryConfig& memory_config = ttnn::DRAM_MEMORY_CONFIG) {
    return tt::numpy::full(shape.with_tile_padding().value(), value, data_type, layout, &device, memory_config);
}

inline ttnn::Tensor zeros(
    const ttnn::Shape& shape,
    const DataType data_type,
    const Layout layout,
    Device& device,
    const MemoryConfig& memory_config = ttnn::DRAM_MEMORY_CONFIG) {
    return full(shape, 0.0f, data_type, layout, device, memory_config);
}

inline ttnn::Tensor ones(
    const ttnn::Shape& shape,
    const DataType data_type,
    const Layout layout,
    Device& device,
    const MemoryConfig& memory_config = ttnn::DRAM_MEMORY_CONFIG) {
    return full(shape, 1.0f, data_type, layout, device, memory_config);
}

}  // namespace creation
}  // namespace operations

using operations::creation::full;
using operations::creation::ones;
using operations::creation::zeros;

}  // namespace ttnn
