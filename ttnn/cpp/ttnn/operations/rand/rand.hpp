// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/decorators.hpp"

namespace ttnn::operations::rand {
struct Rand {
    static Tensor invoke(
        QueueId queue_id,
        const ttnn::Shape& size,
        MeshDevice& device,
        const DataType dtype = DataType::BFLOAT16,
        const Layout layout = Layout::TILE,
        const MemoryConfig& memory_config = types::DRAM_MEMORY_CONFIG);

    static Tensor invoke(
        QueueId queue_id,
        const ttnn::Shape& size,
        const DataType dtype = DataType::BFLOAT16,
        const Layout layout = Layout::TILE);
};
}  // namespace ttnn::operations::rand

namespace ttnn {
constexpr auto rand = ttnn::register_operation<"ttnn::rand", ttnn::operations::rand::Rand>();
}  // namespace ttnn
