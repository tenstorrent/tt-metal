
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
    const std::optional<DataType> dtype,
    const std::optional<Layout> layout,
    std::optional<std::reference_wrapper<MeshDevice>> device,
    const std::optional<MemoryConfig>& memory_config) {
    TT_FATAL(dtype.has_value(), "Missing 'dtype': argument not set and no default value available.");
    TT_FATAL(layout.has_value(), "Missing 'layout': argument not set and no default value available.");

    auto output = ttnn::random::random(ttnn::Shape{size}, *dtype, *layout);
    if (device.has_value()) {
        TT_FATAL(
            memory_config.has_value(),
            "Missing 'memory_config': argument not set and no default value available. This is required when used with "
            "the 'device' argument.");
        return output.to_device(std::addressof(device->get()), *memory_config);
    }
    return output;
}
}  // namespace ttnn::operations::rand
