
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "rand.hpp"

#include "ttnn/operations/rand/device/rand_device_operation.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::rand {

Tensor Rand::invoke(
    const ttnn::Shape& shape,
    MeshDevice& device,
    const DataType dtype,
    const Layout layout,
    const MemoryConfig& memory_config,
    float from,
    float to,
    uint32_t seed) {
    TT_FATAL(dtype != DataType::UINT8, "[ttnn::rand] DataType::UINT8 is not supported.");

    fprintf(stderr, "-- Rand::invoke: shape rank %zu volume %lu [ ", shape.rank(), shape.volume());
    for (size_t i = 0; i < shape.rank(); i++) {
        fprintf(stderr, "%u ", shape[i]);
    }
    fprintf(stderr, "]\n");
    auto tensor = ttnn::prim::uniform(shape, DataType::FLOAT32, Layout::TILE, memory_config, device, from, to, seed);
    if (dtype != DataType::FLOAT32) {
        tensor = ttnn::typecast(tensor, dtype);
    }
    if (layout != Layout::TILE) {
        tensor = ttnn::to_layout(tensor, layout);
    }
    return tensor;
}
}  // namespace ttnn::operations::rand
