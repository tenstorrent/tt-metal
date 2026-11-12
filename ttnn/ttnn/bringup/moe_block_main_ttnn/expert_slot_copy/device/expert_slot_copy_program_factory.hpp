// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "expert_slot_copy_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include "../../common/address_patches.hpp"

namespace ttnn::operations::bringup::moe_block_main::expert_slot_copy {

// One program a chip: on each port core (one a direction and link) a reader (RISCV_1, NoC0) and a writer (RISCV_0,
// NoC1, the port's fabric connection). Buffer addresses bind as Buffer* (address_patches.hpp); the jobs are runtime
// args rewritten on every call, so one cached program serves every (source, destination) pair.
struct SlotCopyProgramFactory {
    struct shared_variables_t {
        AddressPatches patches;
        std::vector<tt::tt_metal::CoreCoord> cores;  // the chip's port cores, in the order build_jobs fills them
    };
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const SlotCopyParams& args,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const SlotCopyInputs& tensor_args,
        Tensor& output);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached, const SlotCopyParams& args, const SlotCopyInputs& tensor_args, Tensor& output);
};

}  // namespace ttnn::operations::bringup::moe_block_main::expert_slot_copy
