
// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rand.hpp"

#include "ttnn/operations/functions.hpp"

namespace ttnn::operations::rand {
Tensor Rand::invoke(
    QueueId queue_id,
    const ttnn::Shape& size,
    std::optional<std::reference_wrapper<MeshDevice>> device,
    const DataType dtype,
    const Layout layout,
    const MemoryConfig& memory_config) {
    auto output = ttnn::random::random(ttnn::Shape{size}, dtype, layout);
    if (device.has_value()) {
        return output.to_device(std::addressof(device->get()), memory_config);
    }
    return output;
}
}  // namespace ttnn::operations::rand
