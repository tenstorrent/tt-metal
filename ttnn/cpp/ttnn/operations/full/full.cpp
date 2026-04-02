// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "full.hpp"

#include "device/full_device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/graph/composite_trace.hpp"

namespace ttnn {

Tensor moreh_full(
    const ttnn::SmallVector<uint32_t>& shape,
    const std::variant<float, int> fill_value,
    ttnn::MeshDevice* mesh_device,
    const DataType& dtype,
    const Layout& layout,
    const MemoryConfig& memory_config) {
    ttnn::graph::ScopedCompositeTrace _trace("ttnn::moreh_full");
    return ttnn::prim::full(shape, fill_value, mesh_device, dtype, layout, memory_config);
}

}  // namespace ttnn
