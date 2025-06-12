
// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rand.hpp"
#include <memory>

#include "tt-metalium/assert.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/functions.hpp"

namespace ttnn::operations::rand {
Tensor Rand::invoke(
    const std::vector<uint32_t>& size,
    std::optional<DataType> dtype,
    std::optional<Layout> layout,
    std::optional<std::reference_wrapper<MeshDevice>> device,
    const std::optional<MemoryConfig>& memory_config) {
    TT_FATAL(dtype.has_value(), "Expected 'dtype' to be set but found no value");
    TT_FATAL(layout.has_value(), "Expected 'layout' to be set but found no value");

    auto output = ttnn::random::random(ttnn::Shape{size}, *dtype, *layout);
    if (device.has_value()) {
        TT_FATAL(memory_config.has_value(), "Expected 'memory_config' to be set but found no value");
        return output.to_device(std::addressof(device->get()), *memory_config);
    }
    return output;
}
}  // namespace ttnn::operations::rand
