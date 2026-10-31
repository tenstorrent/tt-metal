// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/workload_descriptor.hpp>

#include "routed_experts_combine_device_operation_types.hpp"
#include "ttnn/distributed/types.hpp"

namespace ttnn::operations::experimental::moe_block::routed_experts_combine {

// The flat routed experts and the sparse combine in ONE program per chip (every chip's program differs: its ports,
// its accumulators). Every address the programs bake is a tensor base (BufferBinding, patched on a cache hit) or an
// offset from the program's arena CB; the hash holds no address.
struct RoutedExpertsCombineFactory {
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const RoutedExpertsCombineParams& params,
        const RoutedExpertsCombineInputs& inputs,
        ttnn::Tensor& output,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

}  // namespace ttnn::operations::experimental::moe_block::routed_experts_combine
