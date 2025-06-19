
// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rand.hpp"

#include "ttnn/operations/functions.hpp"

namespace ttnn::operations::rand {

Tensor Rand::invoke(QueueId queue_id, const ttnn::Shape& size, const DataType dtype, const Layout layout) {
    return ttnn::random::random(ttnn::Shape{size}, dtype, layout);
}

Tensor Rand::invoke(
    QueueId queue_id,
    const ttnn::Shape& size,
    MeshDevice& device,
    const DataType dtype,
    const Layout layout,
    const MemoryConfig& memory_config) {
    auto output = ttnn::random::random(ttnn::Shape{size}, dtype, layout);
    return output.to_device(std::addressof(device), memory_config);
}
}  // namespace ttnn::operations::rand
