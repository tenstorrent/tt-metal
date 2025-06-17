// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/decorators.hpp"

namespace ttnn::operations::rand {
struct Rand {
    static Tensor invoke(
        const std::vector<uint32_t>& size,
        std::optional<std::reference_wrapper<MeshDevice>> device = std::nullopt,
        const DataType dtype = DataType::BFLOAT16,
        const Layout layout = Layout::TILE,
        const MemoryConfig& memory_config = types::DRAM_MEMORY_CONFIG);
};
}  // namespace ttnn::operations::rand

namespace ttnn {
constexpr auto rand = ttnn::register_operation<"ttnn::rand", ttnn::operations::rand::Rand>();
}  // namespace ttnn
