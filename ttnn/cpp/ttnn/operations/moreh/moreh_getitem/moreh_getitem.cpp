// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_getitem.hpp"

namespace ttnn::operations::moreh::moreh_getitem {
Tensor MorehGetItem::invoke(
    const std::optional<const Tensor>& input,
    const std::vector<Tensor>& index_tensors,
    const ttnn::SmallVector<uint32_t>& index_dims,
    const std::optional<Tensor>& output,
    // const CoreRange core_range,
    const std::optional<MemoryConfig>& memory_config) {
    if (!input.has_value()) {
        // FIXME: This is a hack to work around limitations in the decorator
        // infra which requires either an input tensor or a vector of input
        // tensors but not both; wrapping the input tensor in an optional allows
        // us to work around this without rewriting half of the runtime.
        TT_THROW("Input tensor is required for moreh_getitem operation.");
    }
    return ttnn::prim::moreh_getitem(input.value(), index_tensors, index_dims, output, memory_config);
}
}  // namespace ttnn::operations::moreh::moreh_getitem
