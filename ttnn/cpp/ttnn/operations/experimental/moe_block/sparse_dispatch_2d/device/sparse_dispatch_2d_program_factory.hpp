// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sparse_dispatch_2d_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/experimental/moe_block/common/address_patches.hpp"

namespace ttnn::operations::experimental::moe_block::sparse_dispatch_2d {

// One program a chip (its ports differ): per port core a reader (RISCV_1, NoC0) and a sender (RISCV_0, NoC1, the
// port's fabric link). Every address in the runtime args is a tensor's base (xbuf, route, sems); a cache hit
// rewrites a tensor's arguments only when its address moved (address_patches.hpp). Nothing else is hash-excluded.
struct SparseDispatch2dProgramFactory {
    struct shared_variables_t {
        AddressPatches patches;
    };
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const SparseDispatch2dParams& args,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const SparseDispatch2dInputs& tensor_args,
        Tensor& output);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached,
        const SparseDispatch2dParams& args,
        const SparseDispatch2dInputs& tensor_args,
        Tensor& output);
};

}  // namespace ttnn::operations::experimental::moe_block::sparse_dispatch_2d
