
// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rand.hpp"
#include <type_traits>

#include "ttnn/tensor/layout/page_config.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"

#include "ttnn/operations/rand/device/rand_device_operation.hpp"

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
    return ttnn::prim::uniform(shape, dtype, layout, memory_config, device, from, to, seed);
}
}  // namespace ttnn::operations::rand
