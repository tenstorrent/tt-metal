// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_getitem.hpp"

namespace ttnn::operations::moreh::moreh_getitem {
Tensor MorehGetItem::invoke(
    const Tensor& input,
    const std::vector<Tensor>& index_tensors,
    const std::vector<uint32_t> index_dims,
    const std::optional<Tensor>& output,
    // const CoreRange core_range,
    const std::optional<MemoryConfig> memory_config) {
    return ttnn::prim::moreh_getitem(input, index_tensors, index_dims, output, memory_config);
}
}  // namespace ttnn::operations::moreh::moreh_getitem
