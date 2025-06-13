// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/decorators.hpp"

namespace ttnn::operations::rand {
struct Rand {
    static Tensor invoke(
        const std::vector<uint32_t>& size,
        const std::optional<DataType> dtype,
        const std::optional<Layout> layout,
        std::optional<std::reference_wrapper<MeshDevice>> device,
        const std::optional<MemoryConfig>& memory_config);
};
}  // namespace ttnn::operations::rand

namespace ttnn {
constexpr auto rand = ttnn::register_operation<"ttnn::rand", ttnn::operations::rand::Rand>();
}  // namespace ttnn
