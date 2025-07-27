
// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rand.hpp"

#include "ttnn/operations/rand/device/rand_device_operation.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::rand {

Tensor Rand::invoke(
    QueueId queue_id,
    const ttnn::Shape& shape,
    MeshDevice& device,
    const DataType dtype,
    const Layout layout,
    const MemoryConfig& memory_config,
    float from,
    float to,
    uint32_t seed) {
    TT_FATAL(dtype != DataType::UINT8, "[ttnn::rand] DataType::UINT8 is not supported.");

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
