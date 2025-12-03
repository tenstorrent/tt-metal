// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn::operations::full {
struct Full {
    static ttnn::Tensor invoke(
        const ttnn::SmallVector<uint32_t>& shape,
        std::variant<float, int> fill_value,
        ttnn::MeshDevice* mesh_device,
        const DataType& dtype,
        const Layout& layout,
        const MemoryConfig& memory_config);
};
}  // namespace ttnn::operations::full

namespace ttnn {
constexpr auto moreh_full = ttnn::register_operation<"ttnn::moreh_full", ttnn::operations::full::Full>();
}  // namespace ttnn
