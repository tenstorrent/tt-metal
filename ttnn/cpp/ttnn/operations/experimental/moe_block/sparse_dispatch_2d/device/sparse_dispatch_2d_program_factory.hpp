// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/workload_descriptor.hpp>

#include "sparse_dispatch_2d_device_operation_types.hpp"
#include "ttnn/distributed/types.hpp"

namespace ttnn::operations::experimental::moe_block::sparse_dispatch_2d {

// One program a chip (its ports differ): per port core a reader (RISCV_1, NoC0) and a sender (RISCV_0, NoC1, the
// port's fabric link). Every address in the runtime args is a tensor's base, bound as a Buffer*, so a cache hit only
// patches those bindings; nothing else is hash-excluded (no override_runtime_arguments needed).
struct SparseDispatch2dProgramFactory {
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const SparseDispatch2dParams& args,
        const SparseDispatch2dInputs& tensor_args,
        Tensor& output,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

}  // namespace ttnn::operations::experimental::moe_block::sparse_dispatch_2d
