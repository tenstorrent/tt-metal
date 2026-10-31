// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/workload_descriptor.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"

#include "fused_tail_device_operation_types.hpp"

namespace ttnn::operations::experimental::moe_block::fused_tail {

// One program a chip. Every address a kernel takes is a tensor's base, bound with emplace_runtime_args(Buffer*): a
// cache hit only patches those; the landing / ring / exchange slots other cores and chips write are CBs at the same
// address on every chip, read by the kernels with get_write_ptr.
struct TailEngineFactory {
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const FusedTailParams& attrs,
        const FusedTailInputs& tensor_args,
        Tensor& output,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

struct TailA2AFactory {
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const FusedTailParams& attrs,
        const FusedTailInputs& tensor_args,
        Tensor& output,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

}  // namespace ttnn::operations::experimental::moe_block::fused_tail
