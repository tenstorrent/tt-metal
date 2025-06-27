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
        DataType dtype = DataType::BFLOAT16,
        Layout layout = Layout::TILE,
        const MemoryConfig& memory_config = types::DRAM_MEMORY_CONFIG,
        float from = 0.0f,
        float to = 1.0f,
        uint32_t seed = 0);
};
}  // namespace ttnn::operations::rand

namespace ttnn {
constexpr auto rand = ttnn::register_operation<"ttnn::rand", ttnn::operations::rand::Rand>();
}  // namespace ttnn
