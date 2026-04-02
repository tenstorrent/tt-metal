// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_arange.hpp"

#include "device/moreh_arange_device_operation.hpp"
#include "ttnn/graph/composite_trace.hpp"

namespace ttnn {

Tensor moreh_arange(
    float start,
    float end,
    float step,
    ttnn::MeshDevice* mesh_device,
    const std::optional<Tensor>& output,
    bool untilize_out,
    const DataType& dtype,
    const MemoryConfig& memory_config) {
    ttnn::graph::ScopedCompositeTrace _trace("ttnn::moreh_arange");
    return ttnn::prim::moreh_arange(start, end, step, mesh_device, output, untilize_out, dtype, memory_config);
}

}  // namespace ttnn
